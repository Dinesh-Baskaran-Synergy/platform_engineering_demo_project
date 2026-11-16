"""LiteLLM client tests — all offline (mock_response / monkeypatched litellm.completion)."""
from __future__ import annotations

import litellm
import pytest

from app import llm_client
from app.llm_client import (ConfigError, LiteLLMClient, LLMConfig, MockLLMClient, build_model_response,
                            compute_cost, get_client, normalize_tool_calls)


def test_drop_params_enabled() -> None:
    assert litellm.drop_params is True


def test_mock_mode_defaults_need_no_env() -> None:
    cfg = LLMConfig.from_env(mode="mock", load_dotenv_file=False)
    assert cfg.mode == "mock" and cfg.app_model and cfg.judge_model
    assert isinstance(get_client(cfg), MockLLMClient)


def test_live_mode_requires_models() -> None:
    with pytest.raises(ConfigError):
        LLMConfig.from_env(mode="live", load_dotenv_file=False)


def test_live_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MODEL", "openai/app-alias")
    monkeypatch.setenv("JUDGE_MODEL", "openai/judge-alias")
    monkeypatch.setenv("LITELLM_API_BASE", "http://proxy.invalid")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test-secret-123456")
    cfg = LLMConfig.from_env(mode="live", load_dotenv_file=False)
    assert cfg.extractor_model == "openai/app-alias"  # falls back to APP_MODEL
    desc = cfg.describe()
    assert "sk-test-secret-123456" not in str(desc) and "proxy.invalid" not in str(desc)
    assert desc["connection"] == "proxy" and desc["api_key_set"] is True
    assert "sk-test-secret-123456" not in repr(cfg)


def test_cli_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MODEL", "openai/a")
    monkeypatch.setenv("JUDGE_MODEL", "openai/j")
    cfg = LLMConfig.from_env(mode="live", app_model="anthropic/b", load_dotenv_file=False)
    assert cfg.app_model == "anthropic/b" and cfg.extractor_model == "anthropic/b"


def test_litellm_mock_response_text() -> None:
    resp = litellm.completion(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="hello")
    assert resp.choices[0].message.content == "hello"
    cost, source = compute_cost(resp)
    assert source == "litellm" and cost is not None and cost > 0


def test_cost_unavailable_for_unmapped_alias() -> None:
    resp = litellm.completion(model="openai/my-private-alias", messages=[{"role": "user", "content": "hi"}], mock_response="x")
    assert compute_cost(resp) == (None, "unavailable")


def test_mock_client_tool_calls_shape() -> None:
    client = MockLLMClient()
    from app.tools import TOOL_SCHEMAS

    res = client.complete([{"role": "system", "content": ""}, {"role": "user", "content": "Where is my order ORD-1001?"}],
                          model="openai/mock-acme-app", tools=TOOL_SCHEMAS)
    assert res.tool_calls and res.tool_calls[0]["function"]["name"] == "get_order_status"
    assert res.tool_calls[0]["type"] == "function" and isinstance(res.tool_calls[0]["function"]["arguments"], str)
    assert res.prompt_tokens > 0 and res.completion_tokens > 0
    assert res.cost_usd is None and res.cost_source == "unavailable"


def test_build_model_response_usage() -> None:
    mr = build_model_response("openai/x", "hello world", None, [{"role": "user", "content": "abcd" * 10}])
    assert mr.usage.prompt_tokens == 10 and mr.choices[0].finish_reason == "stop"


def test_normalize_tool_calls_handles_dicts() -> None:
    class Msg:
        tool_calls = [{"id": "1", "function": {"name": "f", "arguments": {"a": 1}}}]

    assert normalize_tool_calls(Msg()) == [{"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}
    real = litellm.completion

    def fake(**kwargs):
        seen.update(kwargs)
        kw = {k: v for k, v in kwargs.items() if k not in {"api_base", "api_key", "fallbacks", "num_retries", "timeout"}}
        return real(**kw, mock_response="pong")

    monkeypatch.setattr(llm_client.litellm, "completion", fake)
    return seen


def test_live_client_passes_proxy_and_resilience_params(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    cfg = LLMConfig(mode="live", api_base="http://proxy.invalid", api_key="sk-abc123456789", app_model="openai/app",
                    judge_model="openai/judge", fallback_model="openai/backup", num_retries=5, timeout=12)
    res = LiteLLMClient(cfg).complete([{"role": "user", "content": "ping"}], model="openai/app", max_tokens=5)
    assert res.content == "pong"
    assert seen["api_base"] == "http://proxy.invalid" and seen["api_key"] == "sk-abc123456789"
    assert seen["num_retries"] == 5 and seen["timeout"] == 12 and seen["fallbacks"] == ["openai/backup"]
    assert seen["temperature"] == 0.0 and seen["max_tokens"] == 5


def test_direct_mode_omits_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    cfg = LLMConfig(mode="live", app_model="anthropic/some-model", judge_model="openai/j")
    LiteLLMClient(cfg).complete([{"role": "user", "content": "x"}], model="anthropic/some-model")
    assert "api_base" not in seen and "api_key" not in seen and "fallbacks" not in seen


def test_errors_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-very-secret-value-999")

    def boom(**kwargs):
        raise ValueError("401 Unauthorized. Authorization: Bearer sk-very-secret-value-999")

    monkeypatch.setattr(llm_client.litellm, "completion", boom)
    cfg = LLMConfig(mode="live", api_key="sk-very-secret-value-999", app_model="openai/a", judge_model="openai/j")
    with pytest.raises(RuntimeError) as exc:
        LiteLLMClient(cfg).complete([{"role": "user", "content": "x"}], model="openai/a")
    assert "sk-very-secret-value-999" not in str(exc.value)


def test_capability_overrides() -> None:
    cfg = LLMConfig(mode="live", api_base="http://p", app_model="openai/unknown-alias-xyz", judge_model="openai/j")
    client = LiteLLMClient(cfg)
    assert client.supports_tools("openai/unknown-alias-xyz") is True  # proxy alias assumed OpenAI-compatible
    cfg.tool_calling = "prompt"
    assert client.supports_tools("openai/unknown-alias-xyz") is False
    cfg.structured_output = "schema"
    assert client.supports_response_schema("openai/unknown-alias-xyz") is True
