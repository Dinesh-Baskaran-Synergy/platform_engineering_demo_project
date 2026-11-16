"""Structured extraction: customer email -> validated JSON ticket fields.

Strategy:
1. If the model supports JSON-schema ``response_format`` (``litellm.supports_response_schema``),
   ask for strict schema output.
2. Otherwise prompt for JSON only.
Either way the output is parsed and validated with pydantic, with ONE repair retry
that feeds the validation error back to the model.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.llm_client import LLMClient, LLMResult
from app.prompt_loader import load_prompt

log = logging.getLogger("acme.extractor")

Intent = Literal["billing", "refund", "technical_issue", "account_access", "order_status",
                 "cancellation", "feedback", "other"]
Urgency = Literal["low", "medium", "high", "critical"]
Product = Literal["compute", "storage", "database", "networking", "api", "billing_portal", "acmeedge", "other"]
Sentiment = Literal["positive", "neutral", "negative"]


class TicketFields(BaseModel):
    """The extraction schema."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    urgency: Urgency
    product: Product
    order_id: str | None
    sentiment: Sentiment


def json_schema() -> dict[str, Any]:
    """Strict JSON schema for ``TicketFields`` (used for response_format and the json_schema grader)."""
    schema = TicketFields.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(TicketFields.model_fields)
    schema["properties"]["order_id"] = {"anyOf": [{"type": "string", "pattern": "^ORD-\\d{4}$"}, {"type": "null"}]}
    return schema


@dataclass
class ExtractionResult:
    """Outcome of one extraction, including all LLM calls made."""

    data: dict[str, Any] | None
    valid: bool
    attempts: int
    raw: str | None
    error: str | None = None
    used_response_schema: bool = False
    calls: list[LLMResult] = field(default_factory=list)


def _parse_json(text: str | None) -> Any:
    """Parse JSON, tolerating ```json fences or leading prose."""
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object found")
        t = t[start:end + 1]
    return json.loads(t)


class Extractor:
    """Email -> ``TicketFields`` via an ``LLMClient``."""

    def __init__(self, client: LLMClient, model: str | None = None) -> None:
        self.client = client
        self.model = model or client.config.extractor_model
        self.prompt = load_prompt("extractor_system")

    def extract(self, email: str) -> ExtractionResult:
        """Extract ticket fields from ``email`` (one repair retry on invalid output)."""
        use_schema = self.client.supports_response_schema(self.model)
        rf = {"type": "json_schema", "json_schema": {"name": "ticket_fields", "schema": json_schema(), "strict": True}} if use_schema else None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompt.text},
            {"role": "user", "content": email if email.strip() else "(empty email)"},
        ]
        calls: list[LLMResult] = []
        raw: str | None = None
        error: str | None = None
        for attempt in (1, 2):
            res = self.client.complete(messages, model=self.model, response_format=rf, purpose="extract")
            calls.append(res)
            raw = res.content
            try:
                data = TicketFields.model_validate(_parse_json(raw)).model_dump()
                return ExtractionResult(data, True, attempt, raw, None, use_schema, calls)
            except (ValueError, ValidationError) as exc:
                error = str(exc).splitlines()[0][:300]
                log.warning("extraction attempt %d invalid: %s", attempt, error)
                messages = messages + [
                    {"role": "assistant", "content": raw or ""},
                    {"role": "user", "content": f"Your output was invalid ({error}). Return ONLY the corrected JSON object."},
                ]
        return ExtractionResult(None, False, 2, raw, error, use_schema, calls)
