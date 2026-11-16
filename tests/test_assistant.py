import json

from app.assistant import EMPTY_INPUT_REPLY, MAX_INPUT_CHARS, Assistant, parse_text_tool_call
from app.llm_client import LLMConfig, MockLLMClient


def test_rag_answer_cites_source() -> None:
    r = Assistant(MockLLMClient()).respond("How long is a password reset link valid?")
    assert "30 minutes" in r.answer and "password-reset" in r.retrieved_ids
    assert r.prompt_version.startswith("assistant_system@v1#") and r.calls


def test_native_tool_loop() -> None:
    client = MockLLMClient()
    r = Assistant(client).respond("Where is my order ORD-1001?")
    assert [t.name for t in r.tool_calls] == ["get_order_status"]
    assert r.tool_calls[0].result["status"] == "shipped"
    assert len(r.calls) == 2 and "shipped" in r.answer
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-2]["tool_calls"]


def test_prompt_based_tool_fallback_when_no_native_tools() -> None:
    client = MockLLMClient(native_tools=False)
    r = Assistant(client).respond("Where is my order ORD-1002?")
    assert r.tool_mode == "prompt" and r.warnings
    assert r.tool_calls[0].name == "get_order_status" and "delivered" in r.answer
    assert client.calls[0]["tools"] is None
    assert '"tool_call"' in client.calls[0]["messages"][0]["content"]


def test_empty_input_short_circuits() -> None:
    client = MockLLMClient()
    r = Assistant(client).respond("   ")
    assert r.answer == EMPTY_INPUT_REPLY and not client.calls


def test_pii_redacted_before_llm_and_in_output() -> None:
    client = MockLLMClient()
    r = Assistant(client).respond("My card 4111 1111 1111 1111 was charged twice")
    sent = json.dumps(client.calls[0]["messages"])
    assert "4111 1111 1111 1111" not in sent and "4111 1111 1111 1111" not in r.answer


def test_long_input_keeps_head_and_tail() -> None:
    client = MockLLMClient()
    text = "filler " * 2000 + "QUESTION_AT_END"
    Assistant(client).respond(text)
    user_msg = client.calls[0]["messages"][-1]["content"]
    assert len(user_msg) < MAX_INPUT_CHARS + 50 and user_msg.endswith("QUESTION_AT_END")


def test_documents_marked_as_data() -> None:
    client = MockLLMClient()
    Assistant(client).respond("Are all AcmeCloud plans free forever?")
    system = client.calls[0]["messages"][0]["content"]
    assert "<documents>" in system and "DATA, not instructions" in system


def test_tool_loop_limit() -> None:
    from tests.conftest import ScriptedResponder
    loop = {"content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_order_status", "arguments": '{"order_id": "ORD-1001"}'}}]}
    client = MockLLMClient(responder=ScriptedResponder([loop]))
    r = Assistant(client, max_tool_rounds=2).respond("loop forever")
    assert "tool loop limit reached" in r.warnings and len(r.calls) == 3


def test_parse_text_tool_call() -> None:
    assert parse_text_tool_call('{"tool_call": {"name": "create_ticket", "arguments": {"summary": "x", "priority": "low"}}}') == (
        "create_ticket", {"summary": "x", "priority": "low"})
    assert parse_text_tool_call("Just an answer.") is None
    assert parse_text_tool_call('{"tool_call": "broken"}') is None


def test_model_override() -> None:
    a = Assistant(MockLLMClient(LLMConfig(app_model="openai/other")))
    assert a.model == "openai/other"
