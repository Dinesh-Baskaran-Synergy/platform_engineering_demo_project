from app.tools import TOOL_SCHEMAS, MockDB, create_ticket, execute_tool, get_order_status


def test_schemas_are_openai_format() -> None:
    for t in TOOL_SCHEMAS:
        assert t["type"] == "function" and {"name", "description", "parameters"} <= set(t["function"])


def test_order_lookup_variants() -> None:
    db = MockDB()
    assert get_order_status(db, "ord-1001")["status"] == "shipped"
    assert get_order_status(db, "ORD-12")["error"] == "invalid_order_id"
    assert get_order_status(db, "ORD-5555")["error"] == "not_found"


def test_tool_never_returns_customer_pii() -> None:
    assert "_customer_email" not in get_order_status(MockDB(), "ORD-1005")


def test_execute_tool_converts_failures() -> None:
    db = MockDB()
    assert execute_tool(db, "get_order_status", '{"order_id": "ORD-9999"}')["error"] == "service_unavailable"
    assert execute_tool(db, "nope", "{}")["error"] == "unknown_tool"
    assert execute_tool(db, "get_order_status", "{bad json")["error"] == "invalid_arguments_json"
    assert execute_tool(db, "get_order_status", {"wrong": 1})["error"] == "bad_arguments"


def test_ticket_ids_deterministic_and_validated() -> None:
    db = MockDB()
    assert create_ticket(db, "db down", "urgent")["ticket_id"] == "TCK-1001"
    assert create_ticket(db, "again", "low")["ticket_id"] == "TCK-1002"
    assert create_ticket(db, "x", "whenever")["error"] == "invalid_priority"
    assert create_ticket(db, " ", "low")["error"] == "missing_summary"
