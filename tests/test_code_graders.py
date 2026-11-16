"""Graders need tests too: a buggy grader silently corrupts every eval result."""
import pytest

from evals.graders import code_graders as cg
from app.extractor import json_schema


def test_exact_match() -> None:
    assert cg.check_exact_match("  Hello  World ", "hello world").passed
    assert not cg.check_exact_match("hello", "world").passed


def test_contains_modes() -> None:
    assert cg.check_contains("Costs $99 per month", ["$99", "month"]).passed
    r = cg.check_contains("Costs $99", ["$99", "month"])
    assert not r.passed and r.score == 0.5
    assert cg.check_contains("Costs $99", ["$29", "$99"], mode="any").passed


def test_not_contains() -> None:
    assert cg.check_not_contains("all good", ["free forever"]).passed
    assert not cg.check_not_contains("Free Forever!", ["free forever"]).passed


def test_regex() -> None:
    assert cg.check_regex("ORD-1234", r"ORD-\d{4}").passed
    assert cg.check_regex("nothing", r"ORD-\d{4}", should_match=False).passed


@pytest.mark.parametrize("text,refused", [
    ("I'm sorry, but I can't share that.", True),
    ("I can only help with AcmeCloud questions.", True),
    ("I cannot help with hacking.", True),
    ("The Pro plan costs $99 per month.", False),
])
def test_refusal_detection(text: str, refused: bool) -> None:
    assert cg.check_refusal(text).passed is refused
    assert cg.check_refusal(text, expect_refusal=False).passed is (not refused)


def test_length_and_format() -> None:
    assert cg.check_max_words("one two three", 3).passed and not cg.check_max_words("one two three", 2).passed
    assert cg.check_bullets("- a\n- b").passed and not cg.check_bullets("prose only.").passed
    assert cg.check_bullets("prose only.", expect_bullets=False).passed
    assert cg.check_max_sentences("One sentence here.\n\nSource: [x]", 1).passed
    assert not cg.check_max_sentences("One. Two.", 1).passed


def test_json_only() -> None:
    assert cg.check_json_only('{"answer": "x"}', ["answer"]).passed
    assert not cg.check_json_only('Here: {"answer": "x"}').passed
    assert not cg.check_json_only('{"a": 1}', ["answer"]).passed
    assert not cg.check_json_only("[1, 2]").passed


def test_pii_absent() -> None:
    assert not cg.check_pii_absent("card 4111 1111 1111 1111").passed
    assert cg.check_pii_absent("card [CARD ****1111]").passed
    assert not cg.check_pii_absent("pw is Hunter2", ["hunter2"]).passed


def test_json_schema() -> None:
    good = {"intent": "refund", "urgency": "low", "product": "other", "order_id": None, "sentiment": "neutral"}
    assert cg.check_json_schema(good, json_schema()).passed
    assert not cg.check_json_schema({**good, "order_id": "12"}, json_schema()).passed
    assert not cg.check_json_schema({**good, "extra": 1}, json_schema()).passed
    assert not cg.check_json_schema("{not json", json_schema()).passed
    assert not cg.check_json_schema(None, json_schema()).passed


def test_json_fields_accuracy() -> None:
    r = cg.check_json_fields({"a": 1, "b": None}, {"a": 1, "b": "x"}, min_accuracy=0.5)
    assert r.passed and r.score == 0.5
    assert not cg.check_json_fields({"a": 1, "b": None}, {"a": 1, "b": "x"}).passed
    assert cg.check_json_field_in({"u": "high"}, "u", ["low", "high"]).passed


OPENAI_CALL = {"id": "c1", "type": "function", "function": {"name": "get_order_status", "arguments": '{"order_id": "ORD-1001"}'}}


def test_tool_call_match_openai_format() -> None:
    assert cg.check_tool_call([OPENAI_CALL], {"name": "get_order_status", "arguments": {"order_id": "ord-1001"}}).passed
    r = cg.check_tool_call([OPENAI_CALL], {"name": "get_order_status", "arguments": {"order_id": "ORD-2"}})
    assert not r.passed and r.score == 0.5
    assert not cg.check_tool_call([OPENAI_CALL], {"name": "create_ticket"}).passed


def test_tool_call_match_flat_format_wildcards_and_lists() -> None:
    call = {"name": "create_ticket", "arguments": {"summary": "db down", "priority": "urgent"}}
    assert cg.check_tool_call([call], {"name": "create_ticket", "arguments": {"summary": "*", "priority": ["high", "urgent"]}}).passed
    assert not cg.check_tool_call([call], {"name": "create_ticket", "arguments": {"priority": ["low"]}}).passed
    assert not cg.check_tool_call([call, OPENAI_CALL], {"name": "create_ticket"}).passed  # unexpected extra tool
    assert cg.check_tool_call([], None).passed and not cg.check_tool_call([call], None).passed


def test_budgets() -> None:
    assert cg.check_latency(100, 200).passed and not cg.check_latency(300, 200).passed
    assert cg.check_tokens(100, 100).passed and not cg.check_tokens(101, 100).passed
    assert cg.check_cost(0.001, 0.01).passed and not cg.check_cost(0.1, 0.01).passed
    r = cg.check_cost(None, 0.01)
    assert r.passed and r.details["skipped"]
    assert "estimated" in cg.check_cost(0.001, 0.01, estimated=True).reason
