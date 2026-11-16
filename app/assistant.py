"""Main orchestration: route -> retrieve -> tools -> answer.

The assistant is deliberately small so every step is visible in eval logs:
retrieved doc ids, each tool call (name, arguments, result) and each LLM call
(latency, tokens, cost) are returned in ``AssistantResponse``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.llm_client import LLMClient, LLMResult
from app.privacy import redact_pii
from app.prompt_loader import load_prompt
from app.retriever import RetrievedDoc, Retriever
from app.text_utils import is_followup
from app.tools import TOOL_SCHEMAS, MockDB, execute_tool, tool_specs_text

log = logging.getLogger("acme.assistant")

MAX_INPUT_CHARS = 4000
EMPTY_INPUT_REPLY = "It looks like your message was empty. How can I help you with AcmeCloud today?"
TOOL_LOOP_REPLY = "Sorry, I couldn't complete that request. Would you like me to create a support ticket?"


@dataclass
class ToolInvocation:
    """One executed tool call."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    id: str | None = None


@dataclass
class AssistantResponse:
    """Everything the evals need to grade one assistant turn."""

    answer: str
    route: str
    retrieved: list[RetrievedDoc] = field(default_factory=list)
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    calls: list[LLMResult] = field(default_factory=list)
    prompt_version: str = ""
    model: str = ""
    tool_mode: str = "native"
    warnings: list[str] = field(default_factory=list)

    @property
    def retrieved_ids(self) -> list[str]:
        """Doc ids in rank order."""
        return [d.doc_id for d in self.retrieved]


def _normalize(conversation: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(conversation, str):
        return [{"role": "user", "content": conversation}]
    return [{"role": m["role"], "content": m.get("content") or ""} for m in conversation if m["role"] in {"user", "assistant"}]


def _documents_block(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "<documents>\n(no relevant documents found)\n</documents>"
    parts = [f'<document id="{d.doc_id}" title="{d.title}">\n{d.text}\n</document>' for d in docs]
    return "<documents>\n" + "\n".join(parts) + "\n</documents>"


def parse_text_tool_call(content: str | None) -> tuple[str, dict[str, Any]] | None:
    """Parse the prompt-based protocol ``{"tool_call": {"name": ..., "arguments": {...}}}``."""
    if not content or "tool_call" not in content:
        return None
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    call = obj.get("tool_call") if isinstance(obj, dict) else None
    if not isinstance(call, dict) or "name" not in call:
        return None
    return call["name"], call.get("arguments") or {}


class Assistant:
    """AcmeCloud support assistant."""

    def __init__(self, client: LLMClient, model: str | None = None, retriever: Retriever | None = None,
                 db: MockDB | None = None, k: int = 3, max_tool_rounds: int = 3) -> None:
        self.client = client
        self.model = model or client.config.app_model
        self.retriever = retriever or Retriever()
        self.db = db or MockDB()
        self.k = k
        self.max_tool_rounds = max_tool_rounds
        self.prompt = load_prompt("assistant_system")

    def route(self, history: list[dict[str, Any]]) -> tuple[str, str]:
        """Return ``(route, retrieval_query)``. Follow-ups are merged with the previous user turn."""
        users = [m["content"] for m in history if m["role"] == "user"]
        last = users[-1] if users else ""
        if not last.strip():
            return "empty", ""
        query = f"{users[-2]} {last}" if len(users) > 1 and is_followup(last) else last
        return "support", query

    def respond(self, conversation: str | list[dict[str, Any]]) -> AssistantResponse:
        """Answer the latest user turn of ``conversation``."""
        history = _normalize(conversation)
        for m in history:
            if m["role"] == "user":
                text = redact_pii(m["content"])
                if len(text) > MAX_INPUT_CHARS:
                    half = MAX_INPUT_CHARS // 2  # keep head and tail: the question is often at the end
                    text = text[:half] + " …[truncated]… " + text[-half:]
                m["content"] = text
        route, query = self.route(history)
        resp = AssistantResponse(answer="", route=route, prompt_version=self.prompt.version, model=self.model)
        if route == "empty":
            resp.answer = EMPTY_INPUT_REPLY
            return resp

        resp.retrieved = self.retriever.retrieve(query, k=self.k)
        native = self.client.supports_tools(self.model)
        system = self.prompt.text + "\n\n" + _documents_block(resp.retrieved)
        if not native:
            resp.tool_mode = "prompt"
            msg = f"Model {self.model} has no native tool calling; using prompt-based tool protocol."
            log.warning(msg)
            resp.warnings.append(msg)
            system += "\n\n" + load_prompt("tool_fallback").text.format(tool_specs=tool_specs_text()).replace("{{", "{").replace("}}", "}")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *history]

        for _ in range(self.max_tool_rounds + 1):
            res = self.client.complete(messages, model=self.model, tools=TOOL_SCHEMAS if native else None, purpose="answer")
            resp.calls.append(res)
            if native and res.tool_calls:
                messages.append({"role": "assistant", "content": res.content, "tool_calls": res.tool_calls})
                for tc in res.tool_calls:
                    result = execute_tool(self.db, tc["function"]["name"], tc["function"]["arguments"])
                    resp.tool_calls.append(ToolInvocation(tc["function"]["name"], _safe_args(tc["function"]["arguments"]), result, tc["id"]))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                continue
            text_call = None if native else parse_text_tool_call(res.content)
            if text_call:
                name, args = text_call
                result = execute_tool(self.db, name, args)
                resp.tool_calls.append(ToolInvocation(name, args, result))
                messages.append({"role": "assistant", "content": res.content})
                messages.append({"role": "user", "content": f'<tool_result name="{name}">{json.dumps(result)}</tool_result>'})
                continue
            resp.answer = redact_pii(res.content or "")
            return resp
        resp.answer = TOOL_LOOP_REPLY
        resp.warnings.append("tool loop limit reached")
        return resp


def _safe_args(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}
