"""Compare a run against the committed baseline and flag regressions.

    python -m evals.compare                                  # results/latest.json vs results/baseline.json
    python -m evals.compare --current results/X.json --baseline results/Y.json

Exit code 1 when any case that passed in the baseline now fails (a regression),
or when a critical baseline case is missing from the current run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
console = Console(width=None if Console().is_terminal else 160)


def load(path: Path) -> dict[str, Any]:
    """Load a results JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def diff_runs(baseline: dict[str, Any], current: dict[str, Any], score_tolerance: float = 0.15) -> dict[str, Any]:
    """Per-case regressions/improvements plus per-category pass-rate deltas."""
    b = {c["id"]: c for c in baseline["cases"]}
    cur = {c["id"]: c for c in current["cases"]}
    regressions, improvements, score_drops = [], [], []
    for cid, c in cur.items():
        old = b.get(cid)
        if old is None:
            continue
        reasons = [f"{g['grader']}: {g['reason']}" for g in c["graders"] if not g["passed"]]
        if old["passed"] and not c["passed"]:
            regressions.append({"id": cid, "category": c["category"], "severity": c["severity"], "reasons": reasons})
        elif not old["passed"] and c["passed"]:
            improvements.append({"id": cid, "category": c["category"]})
        elif c["passed"] and old.get("judge_score") and c.get("judge_score") and old["judge_score"] - c["judge_score"] >= score_tolerance * 5:
            score_drops.append({"id": cid, "from": old["judge_score"], "to": c["judge_score"]})
    removed = [cid for cid in b if cid not in cur]
    added = [cid for cid in cur if cid not in b]
    # Only flag removed critical cases when the current run covered that case's category.
    covered = {c["category"] for c in current["cases"]}
    removed_critical = [cid for cid in removed if b[cid]["severity"] == "critical" and b[cid]["category"] in covered]
    cats = {}
    for cat in sorted(set(baseline["summary"]["categories"]) | set(current["summary"]["categories"])):
        bo = baseline["summary"]["categories"].get(cat, {}).get("pass_rate")
        co = current["summary"]["categories"].get(cat, {}).get("pass_rate")
        cats[cat] = {"baseline": bo, "current": co, "delta": None if bo is None or co is None else round(co - bo, 4)}
    return {"regressions": regressions, "improvements": improvements, "score_drops": score_drops,
            "added": added, "removed": removed, "removed_critical": removed_critical, "categories": cats,
            "baseline_meta": {k: baseline.get(k) for k in ("run_id", "mode", "app_model", "judge_model")},
            "current_meta": {k: current.get(k) for k in ("run_id", "mode", "app_model", "judge_model")}}


def print_diff(d: dict[str, Any]) -> None:
    """Console rendering of ``diff_runs`` output."""
    bm, cm = d["baseline_meta"], d["current_meta"]
    console.print(f"Baseline [bold]{bm['run_id']}[/] ({bm['mode']}, {bm['app_model']})  →  current [bold]{cm['run_id']}[/] ({cm['mode']}, {cm['app_model']})")
    if bm["mode"] != cm["mode"] or bm["app_model"] != cm["app_model"]:
        console.print("[yellow]warning:[/] baseline and current differ in mode/model; differences may not be regressions of the code.")
    t = Table(title="Pass rate by category")
    for col in ("Category", "Baseline", "Current", "Δ"):
        t.add_column(col)
    for cat, v in d["categories"].items():
        fmt = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
        delta = v["delta"]
        dstr = "-" if delta is None else (f"[red]{delta:+.0%}[/]" if delta < 0 else f"[green]{delta:+.0%}[/]" if delta > 0 else "0%")
        t.add_row(cat, fmt(v["baseline"]), fmt(v["current"]), dstr)
    console.print(t)
    if d["regressions"]:
        console.print(f"[bold red]Regressions ({len(d['regressions'])}):[/]")
        for r in d["regressions"]:
            console.print(f"  [red]✗[/] {r['id']} ({r['category']}, {r['severity']})")
            for reason in r["reasons"]:
                console.print(f"      - {reason}", markup=False, highlight=False)
    if d["improvements"]:
        console.print(f"[bold green]Improvements ({len(d['improvements'])}):[/] " + ", ".join(i["id"] for i in d["improvements"]))
    if d["score_drops"]:
        console.print("[yellow]Judge score drops (still passing):[/] " + ", ".join(f"{s['id']} {s['from']}→{s['to']}" for s in d["score_drops"]))
    if d["added"]:
        console.print(f"New cases (not in baseline): {', '.join(d['added'])}")
    if d["removed_critical"]:
        console.print(f"[red]Critical baseline cases missing from this run:[/] {', '.join(d['removed_critical'])}")
    if not d["regressions"] and not d["removed_critical"]:
        console.print("[bold green]No regressions.[/]")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(prog="python -m evals.compare")
    p.add_argument("--baseline", default=str(ROOT / "results" / "baseline.json"))
    p.add_argument("--current", default=str(ROOT / "results" / "latest.json"))
    p.add_argument("--allow-regressions", action="store_true", help="report but exit 0")
    args = p.parse_args(argv)
    for f in (args.baseline, args.current):
        if not Path(f).exists():
            print(f"Missing results file: {f}. Run `make eval` first (and `make baseline` to create a baseline).", file=sys.stderr)
            return 2
    d = diff_runs(load(Path(args.baseline)), load(Path(args.current)))
    print_diff(d)
    failed = bool(d["regressions"] or d["removed_critical"])
    return 1 if failed and not args.allow_regressions else 0


if __name__ == "__main__":
    sys.exit(main())
