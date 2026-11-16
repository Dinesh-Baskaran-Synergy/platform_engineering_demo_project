"""Test setup: force mock mode, strip real credentials, and block all network access."""
from __future__ import annotations

import os
import socket
from typing import Any

import pytest

os.environ["EVAL_MODE"] = "mock"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
for _var in ("LITELLM_API_BASE", "LITELLM_API_KEY", "APP_MODEL", "JUDGE_MODEL", "EXTRACTOR_MODEL", "FALLBACK_MODEL",
             "TOOL_CALLING", "STRUCTURED_OUTPUT"):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to open a network connection fails the test."""
    def guard(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is not allowed in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


class ScriptedResponder:
    """Responder returning queued messages, for testing retry/repair paths."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.seen: list[dict[str, Any]] = []

    def respond(self, messages: list[dict[str, Any]], *, purpose: str = "answer", tools: Any = None) -> dict[str, Any]:
        self.seen.append({"messages": messages, "purpose": purpose, "tools": tools})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def scripted():
    """Factory for ``ScriptedResponder``."""
    return ScriptedResponder
