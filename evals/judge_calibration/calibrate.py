"""Measure agreement between the configured JUDGE_MODEL and human labels.

    python -m evals.judge_calibration.calibrate [--mode live] [--judge-model ...] [--min-agreement 0.8]

Reports pass/fail agreement, Cohen's kappa, mean absolute error of the 1-5 score and
within-1 agreement, and lists every disagreement so you can fix the rubric (or the labels).
Re-run whenever you change JUDGE_MODEL, the judge prompt or a rubric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from app.llm_client import ConfigError, LLMConfig, get_client
from evals.graders.llm_judge import LLMJudge

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
console = Console(width=None if Console().is_terminal else 160)


def load_examples(path: Path = HERE / "labelled_examples.jsonl") -> list[dict[str, Any]]:
    """Load the human-labelled examples."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate agreement metrics from per-example rows (``judge_pass``, ``human_pass``, scores)."""
    valid = [r for r in rows if r["judge_score"] is not None]
    hp = [r["human_pass"] for r in valid]
    jp = [r["judge_pass"] for r in valid]
    n = len(valid)
    return {
        "examples": len(rows), "parsed": n,
        "pass_agreement": round(sum(x == y for x, y in zip(hp, jp)) / n, 4) if n else 0.0,
        "cohen_kappa": round(cohen_kappa(hp, jp), 4),
        "score_mae": round(sum(abs(r["judge_score"] - r["human_score"]) for r in valid) / n, 3) if n else None,
        "within_1": round(sum(abs(r["judge_score"] - r["human_score"]) <= 1 for r in valid) / n, 4) if n else 0.0,
        "false_pass": sum(1 for r in valid if r["judge_pass"] and not r["human_pass"]),
        "false_fail": sum(1 for r in valid if not r["judge_pass"] and r["human_pass"]),
    }


def run(judge: LLMJudge, examples: list[dict[str, Any]], pass_score: int = 4) -> list[dict[str, Any]]:
    """Judge every example and pair it with the human label."""
    rows = []
    for ex in examples:
        res = judge.judge(ex["question"], ex["response"], rubric=ex["rubric"], context=ex.get("context", ""), reference=ex.get("reference", ""))
        score = res.final_score if res.parse_ok else None
        rows.append({"id": ex["id"], "rubric": ex["rubric"], "human_score": ex["human_score"], "human_pass": ex["human_pass"],
                     "judge_score": score, "judge_pass": bool(score is not None and score >= pass_score),
                     "judge_reasoning": res.final_reasoning or res.error, "dimension_scores": res.scores, "note": ex.get("label_notes", "")})
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(prog="python -m evals.judge_calibration.calibrate")
    p.add_argument("--mode", choices=["mock", "live"])
    p.add_argument("--judge-model")
    p.add_argument("--min-agreement", type=float, default=0.0, help="exit 1 if pass agreement is below this")
    args = p.parse_args(argv)
    try:
        cfg = LLMConfig.from_env(mode=args.mode, judge_model=args.judge_model)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    judge = LLMJudge(get_client(cfg), model=cfg.judge_model)
    rows = run(judge, load_examples())
    stats = agreement_stats(rows)
    t = Table(title=f"Judge calibration — {cfg.judge_model} ({cfg.mode})")
    for col in ("Metric", "Value"):
        t.add_column(col)
    for k, v in stats.items():
        t.add_row(k, f"{v:.2%}" if k in ("pass_agreement", "within_1") else str(v))
    console.print(t)
    dis = [r for r in rows if r["judge_pass"] != r["human_pass"]]
    if dis:
        console.print(f"[yellow]Disagreements ({len(dis)}):[/]")
        for r in dis:
            console.print(f"  {r['id']} [{r['rubric']}] human={r['human_score']} judge={r['judge_score']} — {r['note']}", markup=False)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "judge_calibration.json").write_text(json.dumps({"judge_model": cfg.judge_model, "mode": cfg.mode, "stats": stats, "rows": rows}, indent=2), encoding="utf-8")
    md = [f"# Judge calibration — `{cfg.judge_model}` ({cfg.mode})", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in stats.items()]
    md += ["", "| Example | Rubric | Human | Judge | Agree | Note |", "|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['rubric']} | {r['human_score']} | {r['judge_score']} | {'✅' if r['judge_pass'] == r['human_pass'] else '❌'} | {r['note']} |" for r in rows]
    (out / "judge_calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 1 if stats["pass_agreement"] < args.min_agreement else 0


if __name__ == "__main__":
    sys.exit(main())
