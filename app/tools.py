"""Tools exposed to the assistant, with OpenAI-format JSON schemas and a mock in-memory DB."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

ORDER_ID_RE = re.compile(r"^ORD-\d{4}$")
PRIORITIES = ("low", "normal", "high", "urgent")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the status of an AcmeEdge hardware order by its order ID (format ORD-1234).",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-1234"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket for issues that need a human, or when the customer asks for one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-line summary of the issue"},
                    "priority": {"type": "string", "enum": list(PRIORITIES)},
                },
                "required": ["summary", "priority"],
                "additionalProperties": False,
            },
        },
    },
]


def _seed_orders() -> dict[str, dict[str, Any]]:
    return {
        "ORD-1001": {"status": "shipped", "item": "AcmeEdge Mini", "eta": "2026-11-20", "carrier": "DHL"},
        "ORD-1002": {"status": "delivered", "item": "AcmeEdge Pro", "delivered_on": "2026-11-02"},
        "ORD-1003": {"status": "processing", "item": "AcmeEdge Mini", "eta": "2026-11-28"},
        "ORD-1004": {"status": "cancelled", "item": "AcmeEdge Rack Kit"},
        # Customer PII lives in the DB but is never returned by tools.
        "ORD-1005": {"status": "shipped", "item": "AcmeEdge Pro", "eta": "2026-11-19",
                     "_customer_email": "jane.doe@example.com"},
    }


class ToolError(RuntimeError):
    """Raised by a tool for backend failures (simulated outage)."""


@dataclass
class MockDB:
    """In-memory DB; one instance per assistant so eval cases don't interfere."""

    orders: dict[str, dict[str, Any]] = field(default_factory=_seed_orders)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    unavailable_orders: frozenset[str] = frozenset({"ORD-9999"})  # simulates a backend outage


def get_order_status(db: MockDB, order_id: str) -> dict[str, Any]:
    """Return public order fields, or an ``error`` dict for invalid/unknown IDs."""
    oid = (order_id or "").strip().upper()
    if not ORDER_ID_RE.match(oid):
        return {"error": "invalid_order_id", "order_id": order_id}
    if oid in db.unavailable_orders:
        raise ToolError("order service unavailable (simulated)")
    order = db.orders.get(oid)
    if order is None:
        return {"error": "not_found", "order_id": oid}
    return {"order_id": oid, **{k: v for k, v in order.items() if not k.startswith("_")}}


def create_ticket(db: MockDB, summary: str, priority: str) -> dict[str, Any]:
    """Create a ticket with a deterministic ID."""
    if not summary or not summary.strip():
        return {"error": "missing_summary"}
    if priority not in PRIORITIES:
        return {"error": "invalid_priority", "allowed": list(PRIORITIES)}
    ticket = {"ticket_id": f"TCK-{1001 + len(db.tickets)}", "summary": summary.strip(), "priority": priority}
    db.tickets.append(ticket)
    return ticket


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_order_status": get_order_status,
    "create_ticket": create_ticket,
}


def execute_tool(db: MockDB, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Run a tool by name; never raises — failures become ``{"error": ...}`` for the model to handle."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": "unknown_tool", "name": name}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
    except json.JSONDecodeError:
        return {"error": "invalid_arguments_json"}
    try:
        return fn(db, **args)
    except TypeError as exc:
        return {"error": "bad_arguments", "detail": str(exc)}
    except ToolError as exc:
        return {"error": "service_unavailable", "detail": str(exc)}


def tool_specs_text() -> str:
    """Compact description of tools for the prompt-based (non-native) tool protocol."""
    return "\n".join(
        f"- {t['function']['name']}: {t['function']['description']} "
        f"Arguments JSON schema: {json.dumps(t['function']['parameters'])}"
        for t in TOOL_SCHEMAS
    )
