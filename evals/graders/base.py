"""Shared grader types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradeResult:
    """Result of one grader on one case. ``score`` is normalized to 0..1."""

    grader: str
    passed: bool
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GradeContext:
    """Everything a grader may look at for one case."""

    case: Any  # EvalCase
    output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"name","arguments","result"}]
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_text: str = ""
    extraction: dict[str, Any] | None = None
    extraction_raw: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    cost_estimated: bool = False
    judge: Any = None  # LLMJudge | None
