"""Eval case schema + JSONL loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DATASET_DIR = Path(__file__).parent / "datasets"


class EvalCase(BaseModel):
    """One eval case (one line of a JSONL dataset)."""

    id: str
    category: str
    task: Literal["chat", "extract", "retrieve"] = "chat"
    input: str | None = None
    messages: list[dict[str, Any]] | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    graders: list[dict[str, Any]]
    tags: list[str] = Field(default_factory=list)
    severity: Literal["critical", "normal"] = "normal"
    notes: str = ""
    group: str | None = None  # consistency groups: paraphrases that must agree

    @model_validator(mode="after")
    def _check(self) -> "EvalCase":
        if self.input is None and not self.messages:
            raise ValueError(f"{self.id}: needs 'input' or 'messages'")
        if not self.graders:
            raise ValueError(f"{self.id}: needs at least one grader")
        for g in self.graders:
            if "type" not in g:
                raise ValueError(f"{self.id}: grader without 'type': {g}")
        return self

    def conversation(self) -> list[dict[str, Any]]:
        """Messages to send to the assistant (``input`` becomes one user turn)."""
        return self.messages if self.messages else [{"role": "user", "content": self.input or ""}]

    def question_text(self) -> str:
        """Readable rendering of the input, for judges and reports."""
        if self.messages:
            return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in self.messages)
        return self.input or ""


def load_cases(dataset_dir: Path = DATASET_DIR, categories: list[str] | None = None) -> list[EvalCase]:
    """Load every ``*.jsonl`` dataset (optionally filtered by category). Fails loudly on bad lines."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in sorted(dataset_dir.glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                case = EvalCase.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path.name}:{n}: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path.name}:{n}: duplicate case id {case.id}")
            seen.add(case.id)
            if categories is None or case.category in categories:
                cases.append(case)
    return cases
