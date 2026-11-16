"""Maps grader ``type`` strings (used in JSONL datasets) to grader functions.

To add a grader: write ``grade_<name>(spec, ctx) -> GradeResult`` and register it here.
"""
from __future__ import annotations

from typing import Any, Callable

from evals.graders import code_graders as cg
from evals.graders import retrieval_metrics as rm
from evals.graders.base import GradeContext, GradeResult
from evals.graders.llm_judge import grade_llm_judge

Grader = Callable[[dict[str, Any], GradeContext], GradeResult]

GRADERS: dict[str, Grader] = {
    "exact_match": cg.grade_exact_match,
    "contains": cg.grade_contains,
    "contains_any": cg.grade_contains_any,
    "not_contains": cg.grade_not_contains,
    "regex": cg.grade_regex,
    "refusal_detected": cg.grade_refusal,
    "max_words": cg.grade_max_words,
    "bullet_format": cg.grade_bullets,
    "json_only": cg.grade_json_only,
    "max_sentences": cg.grade_max_sentences,
    "json_field_in": cg.grade_json_field_in,
    "pii_absent": cg.grade_pii_absent,
    "json_schema": cg.grade_json_schema,
    "json_fields": cg.grade_json_fields,
    "tool_call_match": cg.grade_tool_call,
    "no_tool_call": cg.grade_no_tool_call,
    "latency_under": cg.grade_latency,
    "tokens_under": cg.grade_tokens,
    "cost_under": cg.grade_cost,
    "recall_at_k": rm.grade_recall_at_k,
    "mrr": rm.grade_mrr,
    "not_retrieved": rm.grade_not_retrieved,
    "llm_judge": grade_llm_judge,
}


def run_grader(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    """Run one grader spec; unknown types and grader crashes become failed results (never abort a run)."""
    fn = GRADERS.get(spec["type"])
    if fn is None:
        return GradeResult(spec["type"], False, 0.0, f"unknown grader type {spec['type']!r}")
    try:
        return fn(spec, ctx)
    except Exception as exc:  # a broken grader must not kill the whole eval run
        return GradeResult(spec["type"], False, 0.0, f"grader error: {type(exc).__name__}: {exc}")
