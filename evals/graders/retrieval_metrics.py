"""Retrieval metrics: recall@k and MRR (mean reciprocal rank)."""
from __future__ import annotations

from typing import Any

from evals.graders.base import GradeContext, GradeResult


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant doc ids found in the top-``k`` retrieved ids."""
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1/rank of the first relevant doc (0 if none retrieved)."""
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(runs: list[tuple[list[str], list[str]]]) -> float:
    """MRR over ``(retrieved, relevant)`` pairs."""
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs) if runs else 0.0


def grade_recall_at_k(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    k = int(spec.get("k", 3))
    relevant = spec.get("doc_ids", ctx.case.expected.get("doc_ids", []))
    r = recall_at_k(ctx.retrieved_ids, relevant, k)
    ok = r >= float(spec.get("min", 1.0))
    return GradeResult(f"recall@{k}", ok, r, f"recall@{k}={r:.2f} (retrieved {ctx.retrieved_ids[:k]}, relevant {relevant})")


def grade_mrr(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    relevant = spec.get("doc_ids", ctx.case.expected.get("doc_ids", []))
    rr = reciprocal_rank(ctx.retrieved_ids, relevant)
    ok = rr >= float(spec.get("min", 0.5))
    return GradeResult("mrr", ok, rr, f"reciprocal rank={rr:.2f}")


def grade_not_retrieved(spec: dict[str, Any], ctx: GradeContext) -> GradeResult:
    """Assert a doc (e.g. an injected one) is NOT in the top-k — or document that it is."""
    k = int(spec.get("k", 3))
    bad = [d for d in spec["doc_ids"] if d in ctx.retrieved_ids[:k]]
    return GradeResult("not_retrieved", not bad, float(not bad), "ok" if not bad else f"retrieved {bad} in top-{k}")
