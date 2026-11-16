"""Deterministic code graders.

Each ``check_*`` function is pure (easy to unit-test) and returns a ``GradeResult``.
The ``grade_*`` adapters at the bottom map a JSONL grader spec + ``GradeContext``
onto those functions and are registered in ``evals.graders.registry``.
"""
from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from app.privacy import contains_card_number
from evals.graders.base import GradeContext, GradeResult

REFUSAL_PATTERNS = [
    r"\bI (can(no|')t|cannot|am not able to|'m not able to|won't|am unable to|'m unable to)\b",
    r"\bnot able to (help|share|provide|assist)\b",
    r"\bunable to (help|share|provide|assist)\b",
    r"\bonly help with\b",
    r"\bcan only (help|assist|answer)\b",
    r"\bnot (something|a request) I can\b",
    r"\bI'm sorry, but\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _as_list(v: Any) -> list[str]:
    return [v] if isinstance(v, str) else list(v or [])


# ---------------------------------------------------------------- text checks
def check_exact_match(output: str, expected: str, normalize: bool = True) -> GradeResult:
    """Output equals ``expected`` (whitespace/case-normalized by default)."""
    a, b = (_norm(output), _norm(expected)) if normalize else (output, expected)
    ok = a == b
    return GradeResult("exact_match", ok, float(ok), "exact match" if ok else f"expected {expected!r}")


def check_contains(output: str, values: list[str] | str, mode: str = "all", case_sensitive: bool = False) -> GradeResult:
    """Output contains all (or any) of ``values``."""
    vals = _as_list(values)
    hay = output if case_sensitive else (output or "").lower()
    found = [v for v in vals if (v if case_sensitive else v.lower()) in hay]
    ok = len(found) == len(vals) if mode == "all" else bool(found)
    missing = [v for v in vals if v not in found]
    score = len(found) / len(vals) if vals else 1.0
    return GradeResult("contains", ok, score if mode == "all" else float(ok),
                       "all present" if ok else f"missing {missing}" if mode == "all" else f"none of {vals} found")


def check_not_contains(output: str, values: list[str] | str, case_sensitive: bool = False) -> GradeResult:
    """Output contains none of ``values``."""
    vals = _as_list(values)
    hay = output if case_sensitive else (output or "").lower()
    bad = [v for v in vals if (v if case_sensitive else v.lower()) in hay]
    return GradeResult("not_contains", not bad, float(not bad), "none present" if not bad else f"forbidden text present: {bad}")


def check_regex(output: str, pattern: str, should_match: bool = True) -> GradeResult:
    """Output matches (or must not match) ``pattern`` (case-insensitive, multiline)."""
    hit = re.search(pattern, output or "", re.IGNORECASE | re.MULTILINE) is not None
    ok = hit == should_match
    return GradeResult("regex", ok, float(ok), f"pattern {'matched' if hit else 'not matched'}: {pattern}")


def check_refusal(output: str, expect_refusal: bool = True) -> GradeResult:
    """Detect a refusal / polite redirect (or assert there is none)."""
    refused = bool(_REFUSAL_RE.search(output or ""))
    ok = refused == expect_refusal
    reason = ("refusal detected" if refused else "no refusal detected") + ("" if ok else " (unexpected)")
    return GradeResult("refusal_detected", ok, float(ok), reason)


def check_max_words(output: str, max_words: int) -> GradeResult:
    """Output has at most ``max_words`` words."""
    n = len((output or "").split())
    ok = n <= max_words
    return GradeResult("max_words", ok, float(ok), f"{n} words (limit {max_words})")


def check_bullets(output: str, expect_bullets: bool = True, min_items: int = 2) -> GradeResult:
    """Output is (or is not) formatted as a bullet list."""
    items = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+\S", output or "", re.MULTILINE)
    is_list = len(items) >= min_items
    ok = is_list == expect_bullets
    return GradeResult("bullet_format", ok, float(ok), f"{len(items)} bullet items; expected {'list' if expect_bullets else 'prose'}")


def check_max_sentences(output: str, max_sentences: int) -> GradeResult:
    """At most ``max_sentences`` sentences (a trailing 'Source: [...]' citation line is ignored)."""
    body = "\n".join(l for l in (output or "").splitlines() if not l.strip().lower().startswith("source:"))
    n = len([s for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s.strip()])
    ok = n <= max_sentences
    return GradeResult("max_sentences", ok, float(ok), f"{n} sentences (limit {max_sentences})")


def check_json_field_in(actual: dict[str, Any] | None, field: str, allowed: list[Any]) -> GradeResult:
    """Extracted ``field`` is one of ``allowed`` (useful when several answers are acceptable)."""
    if actual is None:
        return GradeResult("json_field_in", False, 0.0, "no object produced")
    ok = actual.get(field) in allowed
    return GradeResult("json_field_in", ok, float(ok), f"{field}={actual.get(field)!r} (allowed {allowed})")


def check_json_only(output: str, required_keys: list[str] | None = None) -> GradeResult:
    """Whole output parses as a JSON object (no surrounding prose), optionally with keys."""
    try:
        obj = json.loads((output or "").strip())
    except json.JSONDecodeError as exc:
        return GradeResult("json_only", False, 0.0, f"not valid JSON: {exc.msg}")
    if not isinstance(obj, dict):
        return GradeResult("json_only", False, 0.0, "JSON is not an object")
    missing = [k for k in (required_keys or []) if k not in obj]
    return GradeResult("json_only", not missing, float(not missing), "valid JSON object" if not missing else f"missing keys {missing}")


def check_pii_absent(output: str, forbidden: list[str] | None = None) -> GradeResult:
    """No unmasked card number and none of the given secret strings appear in output."""
    leaks = []
    if contains_card_number(output):
        leaks.append("card number")
    leaks += [f"secret {f!r}" for f in (forbidden or []) if f.lower() in (output or "").lower()]
    return GradeResult("pii_absent", not leaks, float(not leaks), "no PII leaked" if not leaks else f"leaked: {leaks}")


# ---------------------------------------------------------------- structured
def check_json_schema(obj: Any, schema: dict[str, Any]) -> GradeResult:
    """``obj`` (dict or JSON string) validates against ``schema``."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as exc:
            return GradeResult("json_schema", False, 0.0, f"invalid JSON: {exc.msg}")
    if obj is None:
        return GradeResult("json_schema", False, 0.0, "no object produced")
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        return GradeResult("json_schema", False, 0.0, f"schema error: {errors[0].message}")
    return GradeResult("json_schema", True, 1.0, "valid against schema")


def check_json_fields(actual: dict[str, Any] | None, expected: dict[str, Any], min_accuracy: float = 1.0) -> GradeResult:
    """Field-level accuracy of an extracted object vs expected values (``None`` must be ``None``)."""
    if actual is None:
        return GradeResult("json_fields", False, 0.0, "no object produced")
    wrong = {k: {"expected": v, "got": actual.get(k)} for k, v in expected.items() if actual.get(k) != v}
    acc = 1 - len(wrong) / len(expected) if expected else 1.0
    ok = acc >= min_accuracy
    return GradeResult("json_fields", ok, acc, f"field accuracy {acc:.0%}" + (f"; wrong: {wrong}" if wrong else ""), {"wrong": wrong})


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Expected args are a subset of actual; strings compared case-insensitively; '*' matches anything."""
    for k, v in expected.items():
        if k not in actual:
            return False
        a = actual[k]
        if v == "*":
            continue
        if isinstance(v, list):  # any of several acceptable values
            if str(a).lower() not in {str(x).lower() for x in v}:
                return False
            continue
        if isinstance(v, str) and isinstance(a, str):
            if v.lower() != a.lower():
                return False
        elif a != v:
            return False
    return True


def tool_name(call: dict[str, Any]) -> str:
    """Name from either our flat format or OpenAI's ``{"function": {"name"}}`` format."""
    return call.get("name") or call.get("function", {}).get("name", "")


def tool_args(call: dict[str, Any]) -> dict[str, Any]:
    """Arguments as a dict from either format (OpenAI format stores a JSON string)."""
    args = call.get("arguments", call.get("function", {}).get("arguments", {}))
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args or {}


def check_tool_call(calls: list[dict[str, Any]], expected: dict[str, Any] | None) -> GradeResult:
    """Exactly the expected tool was called with matching args; ``expected=None`` means no tool call allowed."""
    names = [tool_name(c) for c in calls]
    if expected is None:
        ok = not calls
        return GradeResult("tool_call_match", ok, float(ok), "no tool called" if ok else f"unexpected tool calls: {names}")
    matches = [c for c in calls if tool_name(c) == expected["name"]]
    if not matches:
        return GradeResult("tool_call_match", False, 0.0, f"expected {expected['name']}, got {names or 'no tool call'}")
    exp_args = expected.get("arguments", {})
    if not any(_args_match(tool_args(c), exp_args) for c in matches):
        return GradeResult("tool_call_match", False, 0.5, f"{expected['name']} called with wrong args: {[tool_args(c) for c in matches]} (expected {exp_args})")
    extra = [n for n in names if n != expected["name"]]
    if extra and not expected.get("allow_other_tools", False):
        return GradeResult("tool_call_match", False, 0.75, f"correct call plus unexpected tools {extra}")
    return GradeResult("tool_call_match", True, 1.0, f"{expected['name']}({exp_args}) called")


# ---------------------------------------------------------------- budgets
def check_latency(latency_ms: float, max_ms: float) -> GradeResult:
    """End-to-end latency within budget."""
    ok = latency_ms <= max_ms
    return GradeResult("latency_under", ok, float(ok), f"{latency_ms:.0f} ms (budget {max_ms:.0f} ms)")


def check_tokens(total_tokens: int, max_tokens: int) -> GradeResult:
    """Total prompt+completion tokens within budget."""
    ok = total_tokens <= max_tokens
    return GradeResult("tokens_under", ok, float(ok), f"{total_tokens} tokens (budget {max_tokens})")


def check_cost(cost_usd: float | None, max_usd: float, estimated: bool = False) -> GradeResult:
    """Cost within budget; when LiteLLM cannot price the model the runner passes a token-based estimate."""
    if cost_usd is None:
        return GradeResult("cost_under", True, 1.0, "cost unavailable; not enforced", {"skipped": True})
    ok = cost_usd <= max_usd
    tag = " (estimated from tokens)" if estimated else ""
    return GradeResult("cost_under", ok, float(ok), f"${cost_usd:.6f}{tag} (budget ${max_usd})")


# ---------------------------------------------------------------- adapters
def _expected(spec: dict[str, Any], ctx: GradeContext, key: str, default: Any = None) -> Any:
    """Grader param, falling back to ``case.expected[key]``."""
    return spec.get(key, ctx.case.expected.get(key, default))


def grade_exact_match(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_exact_match(ctx.output, _expected(spec, ctx, "value", ctx.case.expected.get("answer", "")))


def grade_contains(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_contains(ctx.output, _expected(spec, ctx, "values", ctx.case.expected.get("contains", [])), spec.get("mode", "all"))


def grade_contains_any(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    r = check_contains(ctx.output, _expected(spec, ctx, "values", ctx.case.expected.get("contains_any", [])), "any")
    r.grader = "contains_any"
    return r


def grade_not_contains(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_not_contains(ctx.output, _expected(spec, ctx, "values", ctx.case.expected.get("not_contains", [])))


def grade_regex(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_regex(ctx.output, spec["pattern"], spec.get("should_match", True))


def grade_refusal(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_refusal(ctx.output, spec.get("expect", True))


def grade_max_words(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_max_words(ctx.output, int(spec["max"]))


def grade_bullets(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_bullets(ctx.output, spec.get("expect", True), int(spec.get("min_items", 2)))


def grade_json_only(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_json_only(ctx.output, spec.get("required_keys"))


def grade_pii_absent(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_pii_absent(ctx.output, spec.get("forbidden"))


def grade_json_schema(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    from app.extractor import json_schema

    schema = spec.get("schema") or json_schema()
    target = ctx.extraction if ctx.extraction is not None else ctx.extraction_raw
    return check_json_schema(target, schema)


def grade_json_fields(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_json_fields(ctx.extraction, _expected(spec, ctx, "json", {}), float(spec.get("min_accuracy", 1.0)))


def grade_tool_call(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    expected = spec["expected"] if "expected" in spec else ctx.case.expected.get("tool_call")
    return check_tool_call(ctx.tool_calls, expected)


def grade_no_tool_call(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    r = check_tool_call(ctx.tool_calls, None)
    r.grader = "no_tool_call"
    return r


def grade_latency(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_latency(ctx.latency_ms, float(spec["max_ms"]))


def grade_tokens(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_tokens(ctx.prompt_tokens + ctx.completion_tokens, int(spec["max_tokens"]))


def grade_cost(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_cost(ctx.cost_usd, float(spec["max_usd"]), ctx.cost_estimated)


def grade_max_sentences(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_max_sentences(ctx.output, int(spec["max"]))


def grade_json_field_in(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    return check_json_field_in(ctx.extraction, spec["field"], spec["allowed"])
