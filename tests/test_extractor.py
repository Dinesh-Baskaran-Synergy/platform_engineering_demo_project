import json

from app.extractor import Extractor, TicketFields, json_schema
from app.llm_client import MockLLMClient

GOOD = {"intent": "refund", "urgency": "low", "product": "billing_portal", "order_id": None, "sentiment": "neutral"}


def test_mock_extraction_valid() -> None:
    res = Extractor(MockLLMClient()).extract("Where is my order ORD-1003? It never arrived, really disappointed.")
    assert res.valid and res.data["order_id"] == "ORD-1003" and res.data["intent"] == "order_status"
    TicketFields.model_validate(res.data)


def test_uses_response_schema_when_supported() -> None:
    client = MockLLMClient(response_schema=True)
    res = Extractor(client).extract("hello")
    assert res.used_response_schema and client.calls[0]["response_format"]["type"] == "json_schema"


def test_prompt_json_when_schema_unsupported() -> None:
    client = MockLLMClient(response_schema=False)
    res = Extractor(client).extract("hello")
    assert not res.used_response_schema and client.calls[0]["response_format"] is None


def test_repair_retry(scripted) -> None:
    resp = scripted([{"content": "Sure! Here you go: {intent: refund}", "tool_calls": None},
                     {"content": "```json\n" + json.dumps(GOOD) + "\n```", "tool_calls": None}])
    res = Extractor(MockLLMClient(responder=resp)).extract("refund please")
    assert res.valid and res.attempts == 2 and res.data == GOOD
    assert "invalid" in resp.seen[1]["messages"][-1]["content"].lower()


def test_gives_up_after_one_repair(scripted) -> None:
    bad = {**GOOD, "urgency": "whenever"}
    res = Extractor(MockLLMClient(responder=scripted([{"content": json.dumps(bad), "tool_calls": None}]))).extract("x")
    assert not res.valid and res.attempts == 2 and res.data is None and res.error


def test_schema_is_strict() -> None:
    s = json_schema()
    assert s["additionalProperties"] is False and set(s["required"]) == set(GOOD)
