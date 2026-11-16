"""The ONLY place this project talks to LLMs — always through LiteLLM.

* ``LLMConfig``      – env-driven configuration (models, proxy vs direct, retries).
* ``LLMClient``      – the interface the app and the judge depend on.
* ``LiteLLMClient``  – live calls via ``litellm.completion`` (proxy or direct routing).
* ``MockLLMClient``  – offline, deterministic: still calls ``litellm.completion`` but with
  ``mock_response=`` so responses are shaped exactly like real LiteLLM responses.

Every call returns an ``LLMResult`` with content, OpenAI-format tool_calls, latency,
token usage and cost (``None`` when LiteLLM cannot price the model, e.g. proxy aliases).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

# Use LiteLLM's bundled price map instead of fetching it over the network on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.mock_llm import MockResponder  # noqa: E402
from app.privacy import redact_secrets  # noqa: E402

litellm.drop_params = True  # unsupported params (e.g. temperature on some models) are dropped, not fatal
litellm.suppress_debug_info = True

log = logging.getLogger("acme.llm")

MOCK_APP_MODEL = "openai/mock-acme-app"
MOCK_JUDGE_MODEL = "openai/mock-acme-judge"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _env(name: str) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or None


@dataclass
class LLMConfig:
    """Runtime configuration, loaded from environment variables (and ``.env``)."""

    mode: str = "mock"
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)  # never printed
    app_model: str = MOCK_APP_MODEL
    judge_model: str = MOCK_JUDGE_MODEL
    extractor_model: str = MOCK_APP_MODEL
    fallback_model: str | None = None
    timeout: float = 60.0
    num_retries: int = 3
    tool_calling: str = "auto"       # auto | native | prompt
    structured_output: str = "auto"  # auto | schema | prompt

    @classmethod
    def from_env(
        cls,
        mode: str | None = None,
        app_model: str | None = None,
        judge_model: str | None = None,
        load_dotenv_file: bool = True,
    ) -> "LLMConfig":
        """Build config from env vars; CLI overrides win. Live mode requires model names."""
        if load_dotenv_file:
            load_dotenv(override=False)
        mode = (mode or _env("EVAL_MODE") or "mock").lower()
        if mode not in {"mock", "live"}:
            raise ConfigError(f"EVAL_MODE must be 'mock' or 'live', got {mode!r}")
        app = app_model or _env("APP_MODEL")
        judge = judge_model or _env("JUDGE_MODEL")
        if mode == "live":
            missing = [n for n, v in (("APP_MODEL", app), ("JUDGE_MODEL", judge)) if not v]
            if missing:
                raise ConfigError(f"Live mode needs {', '.join(missing)} (set in .env or pass --app-model/--judge-model)")
        else:
            app = app or MOCK_APP_MODEL
            judge = judge or MOCK_JUDGE_MODEL
        return cls(
            mode=mode,
            api_base=_env("LITELLM_API_BASE"),
            api_key=_env("LITELLM_API_KEY"),
            app_model=app,
            judge_model=judge,
            extractor_model=_env("EXTRACTOR_MODEL") if not app_model and _env("EXTRACTOR_MODEL") else app,
            fallback_model=_env("FALLBACK_MODEL"),
            timeout=float(_env("LLM_TIMEOUT_SECONDS") or 60),
            num_retries=int(_env("LLM_NUM_RETRIES") or 3),
            tool_calling=(_env("TOOL_CALLING") or "auto").lower(),
            structured_output=(_env("STRUCTURED_OUTPUT") or "auto").lower(),
        )

    def describe(self) -> dict[str, Any]:
        """Safe-to-log summary (no secrets, no endpoint URL)."""
        d = asdict(self)
        d.pop("api_key")
        d["api_base"] = "set" if self.api_base else "not set"
        d["api_key_set"] = bool(self.api_key)
        d["connection"] = "proxy" if self.api_base else "direct"
        return d


@dataclass
class LLMResult:
    """Normalized result of one LLM call."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str = "unavailable"  # litellm | unavailable
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens."""
        return self.prompt_tokens + self.completion_tokens


def normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Convert LiteLLM tool_call objects into plain OpenAI-format dicts."""
    calls = getattr(message, "tool_calls", None) or []
    out = []
    for c in calls:
        fn = c.function if hasattr(c, "function") else c["function"]
        name = fn.name if hasattr(fn, "name") else fn["name"]
        args = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
        out.append({
            "id": getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None),
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        })
    return out


def compute_cost(response: Any) -> tuple[float | None, str]:
    """Cost via ``litellm.completion_cost``; ``(None, 'unavailable')`` if the model is not priced."""
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # unmapped proxy alias, custom model, etc. — never crash on cost
        return None, "unavailable"
    if cost is None or (isinstance(cost, float) and math.isnan(cost)):
        return None, "unavailable"
    return float(cost), "litellm"


def _to_result(response: Any, model: str, latency_ms: float) -> LLMResult:
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    cost, source = compute_cost(response)
    return LLMResult(
        content=choice.message.content,
        tool_calls=normalize_tool_calls(choice.message),
        model=getattr(response, "model", None) or model,
        latency_ms=latency_ms,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost_usd=cost,
        cost_source=source,
        finish_reason=getattr(choice, "finish_reason", None),
    )


class LLMClient(ABC):
    """Interface used by the assistant, extractor and judge."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        purpose: str = "answer",
    ) -> LLMResult:
        """Run one chat completion and return a normalized ``LLMResult``."""

    @abstractmethod
    def supports_tools(self, model: str) -> bool:
        """Whether native (OpenAI-format) tool calling should be used for ``model``."""

    @abstractmethod
    def supports_response_schema(self, model: str) -> bool:
        """Whether JSON-schema ``response_format`` should be used for ``model``."""


def capability(fn_name: str, model: str) -> bool | None:
    """Ask LiteLLM about a model capability; ``None`` if LiteLLM does not know the model."""
    try:
        litellm.get_model_info(model=model)  # raises for unmapped models (e.g. proxy aliases)
    except Exception:
        return None
    try:
        return bool(getattr(litellm, fn_name)(model=model))
    except Exception:
        return None


class LiteLLMClient(LLMClient):
    """Real calls through LiteLLM (proxy if LITELLM_API_BASE is set, else direct provider routing)."""

    def _kwargs(self, model: str) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": model,
            "timeout": self.config.timeout,
            "num_retries": self.config.num_retries,  # LiteLLM retries 429/5xx with backoff
        }
        if self.config.api_base:
            kw["api_base"] = self.config.api_base
        if self.config.api_key:
            kw["api_key"] = self.config.api_key
        if self.config.fallback_model and self.config.fallback_model != model:
            kw["fallbacks"] = [self.config.fallback_model]
        return kw

    def complete(self, messages, *, model, tools=None, response_format=None, temperature=0.0,
                 max_tokens=None, purpose="answer") -> LLMResult:
        kw = self._kwargs(model)
        kw.update(messages=messages, temperature=temperature)
        if tools:
            kw["tools"] = tools
        if response_format:
            kw["response_format"] = response_format
        if max_tokens:
            kw["max_tokens"] = max_tokens
        start = time.perf_counter()
        try:
            resp = litellm.completion(**kw)
        except Exception as exc:  # re-raise without leaking keys in the message
            raise RuntimeError(redact_secrets(f"{type(exc).__name__}: {exc}")) from None
        return _to_result(resp, model, (time.perf_counter() - start) * 1000)

    def supports_tools(self, model: str) -> bool:
        if self.config.tool_calling in {"native", "prompt"}:
            return self.config.tool_calling == "native"
        cap = capability("supports_function_calling", model)
        if cap is None:
            # Unknown to LiteLLM (typical for proxy aliases): OpenAI-compatible proxies support tools.
            return bool(self.config.api_base)
        return cap

    def supports_response_schema(self, model: str) -> bool:
        if self.config.structured_output in {"schema", "prompt"}:
            return self.config.structured_output == "schema"
        return bool(capability("supports_response_schema", model))


class MockLLMClient(LLMClient):
    """Offline client: deterministic fixture responses passed through ``litellm.completion(mock_response=...)``."""

    def __init__(self, config: LLMConfig | None = None, responder: MockResponder | None = None,
                 native_tools: bool = True, response_schema: bool = True) -> None:
        super().__init__(config or LLMConfig())
        self.responder = responder or MockResponder()
        self._native_tools = native_tools
        self._response_schema = response_schema
        self.calls: list[dict[str, Any]] = []  # recorded for tests

    def complete(self, messages, *, model, tools=None, response_format=None, temperature=0.0,
                 max_tokens=None, purpose="answer") -> LLMResult:
        self.calls.append({"model": model, "purpose": purpose, "messages": messages, "tools": tools,
                           "response_format": response_format})
        msg = self.responder.respond(messages, purpose=purpose, tools=tools)
        mock = build_model_response(model, msg.get("content"), msg.get("tool_calls"), messages)
        start = time.perf_counter()
        kw: dict[str, Any] = {"model": model, "messages": messages, "mock_response": mock,
                              "temperature": temperature}
        if tools:
            kw["tools"] = tools
        resp = litellm.completion(**kw)
        return _to_result(resp, model, (time.perf_counter() - start) * 1000)

    def supports_tools(self, model: str) -> bool:
        if self.config.tool_calling in {"native", "prompt"}:
            return self.config.tool_calling == "native"
        return self._native_tools

    def supports_response_schema(self, model: str) -> bool:
        if self.config.structured_output in {"schema", "prompt"}:
            return self.config.structured_output == "schema"
        return self._response_schema


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def build_model_response(model: str, content: str | None, tool_calls: list[dict[str, Any]] | None,
                         messages: list[dict[str, Any]]) -> litellm.ModelResponse:
    """Build a LiteLLM ``ModelResponse`` (OpenAI shape, incl. tool_calls and usage) for ``mock_response``."""
    prompt_tokens = sum(_approx_tokens(str(m.get("content") or "")) for m in messages)
    completion_tokens = _approx_tokens((content or "") + json.dumps(tool_calls or []))
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return litellm.ModelResponse(
        model=model,
        choices=[{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
               "total_tokens": prompt_tokens + completion_tokens},
    )


def get_client(config: LLMConfig) -> LLMClient:
    """Factory: ``MockLLMClient`` for mock mode, ``LiteLLMClient`` for live mode."""
    return MockLLMClient(config) if config.mode == "mock" else LiteLLMClient(config)
