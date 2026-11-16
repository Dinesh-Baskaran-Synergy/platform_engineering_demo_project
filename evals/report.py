"""Reporting: rich console tables, ``results/<run_id>.json``, ``results/latest.json`` and ``results/report.md``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console(width=None if Console().is_terminal else 160)


def _fmt_cost(v: float, estimated: bool = False) -> str:
    return f"${v:.4f}{'*' if estimated else ''}"


def _fmt_score(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def failures(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Failed cases with the reasons of their failing graders."""
    out = []
    for r in run["cases"]:
        if r["passed"]:
            continue
        reasons = [f"{g['grader']}: {g['reason']}" for g in r["graders"] if not g["passed"]]
        if r.get("error"):
            reasons.insert(0, f"error: {r['error']}")
        out.append({"id": r["id"], "category": r["category"], "severity": r["severity"], "reasons": reasons, "output": r["output"]})
    return out


def write_run(run: dict[str, Any], out_dir: Path, suffix: str = "", latest: bool = True) -> Path:
    """Persist the run as JSON (+ latest.json) and a Markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run['run_id']}{suffix}.json"
    payload = json.dumps(run, indent=2, ensure_ascii=False, default=str)
    path.write_text(payload, encoding="utf-8")
    if latest:
        (out_dir / "latest.json").write_text(payload, encoding="utf-8")
        (out_dir / "report.md").write_text(render_markdown(run), encoding="utf-8")
    else:
        (out_dir / f"report{suffix}.md").write_text(render_markdown(run), encoding="utf-8")
    return path


def render_markdown(run: dict[str, Any]) -> str:
    """Markdown report: header, per-category table, failures."""
    s, o = run["summary"], run["summary"]["overall"]
    lines = [
        f"# Eval report — {run['run_id']}",
        "",
        f"- Mode: **{run['mode']}** · connection: {run['connection']} · LiteLLM {run.get('litellm_version')}",
        f"- App model: `{run['app_model']}` · Extractor: `{run['extractor_model']}` · Judge: `{run['judge_model']}`",
        f"- Prompts: {', '.join(f'`{v}`' for v in run['prompt_versions'].values())}",
        f"- Git commit: `{run.get('git_commit')}` · duration {run['duration_s']}s",
        f"- **Overall: {o['passed']}/{o['cases']} passed ({o['pass_rate']:.1%}), gate {'PASSED' if o['gate_passed'] else 'FAILED'}** "
        f"(threshold {o['threshold']:.0%}) · mean judge {_fmt_score(o['mean_judge_score'])} · p95 latency {o['p95_latency_ms']:.0f} ms · "
        f"cost {_fmt_cost(o['total_cost_usd'], o['cost_estimated'])}",
    ]
    if o["cost_estimated"]:
        lines.append("- `*` cost estimated from token usage (LiteLLM had no price for this model).")
    for w in run.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
    lines += ["", "| Category | Pass rate | Passed | Threshold | Mean score | Mean judge | p95 latency (ms) | Cost | Status |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cat, c in s["categories"].items():
        status = "✅" if c["met"] else "❌" + (" (critical)" if c["critical_failures"] else "")
        lines.append(f"| {cat} | {c['pass_rate']:.0%} | {c['passed']}/{c['cases']} | {c['threshold']:.0%} | {c['mean_score']:.2f} | "
                     f"{_fmt_score(c['mean_judge_score'])} | {c['p95_latency_ms']:.0f} | {_fmt_cost(c['cost_usd'])} | {status} |")
    fails = failures(run)
    lines += ["", f"## Failures ({len(fails)})", ""]
    if not fails:
        lines.append("None 🎉")
    for f in fails:
        lines.append(f"### `{f['id']}` ({f['category']}, {f['severity']})")
        lines += [f"- {r}" for r in f["reasons"]]
        out = (f["output"] or "").replace("\n", " ")
        lines += [f"- Output: > {out[:400]}", ""]
    return "\n".join(lines) + "\n"


def print_run(run: dict[str, Any]) -> None:
    """Console summary table + failure list."""
    s, o = run["summary"], run["summary"]["overall"]
    t = Table(title=f"AcmeCloud evals — {run['mode']} — app={run['app_model']} judge={run['judge_model']}")
    for col in ("Category", "Pass rate", "Passed", "Thr.", "Mean score", "Judge", "p95 ms", "Cost", ""):
        t.add_column(col, justify="right" if col not in ("Category", "") else "left")
    for cat, c in s["categories"].items():
        t.add_row(cat, f"{c['pass_rate']:.0%}", f"{c['passed']}/{c['cases']}", f"{c['threshold']:.0%}", f"{c['mean_score']:.2f}",
                  _fmt_score(c["mean_judge_score"]), f"{c['p95_latency_ms']:.0f}", _fmt_cost(c["cost_usd"]),
                  "[green]OK[/]" if c["met"] else "[red]FAIL[/]")
    t.add_section()
    t.add_row("[bold]overall", f"[bold]{o['pass_rate']:.0%}", f"{o['passed']}/{o['cases']}", f"{o['threshold']:.0%}", "",
              _fmt_score(o["mean_judge_score"]), f"{o['p95_latency_ms']:.0f}", _fmt_cost(o["total_cost_usd"], o["cost_estimated"]),
              "[green]PASS[/]" if o["gate_passed"] else "[red]FAIL[/]")
    console.print(t)
    for w in run.get("warnings", []):
        console.print(f"[yellow]warning:[/] {w}")
    if o["cost_estimated"]:
        console.print("[dim]* cost estimated from token usage (model not priced by LiteLLM)[/]")
    fails = failures(run)
    if fails:
        console.print(f"\n[bold red]Failures ({len(fails)}):[/]")
        for f in fails:
            console.print(f"  [red]✗[/] [bold]{f['id']}[/] ({f['category']}, {f['severity']})")
            for r in f["reasons"]:
                console.print(f"      - {r}", markup=False, highlight=False)


def comparison_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model for the side-by-side comparison."""
    rows = []
    for r in runs:
        o = r["summary"]["overall"]
        rows.append({"model": r["app_model"], "pass_rate": o["pass_rate"], "passed": f"{o['passed']}/{o['cases']}",
                     "mean_judge_score": o["mean_judge_score"], "p95_latency_ms": o["p95_latency_ms"],
                     "total_cost_usd": o["app_cost_usd"], "cost_estimated": o["cost_estimated"], "gate_passed": o["gate_passed"]})
    return rows


def write_model_comparison(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    """Write ``results/model_comparison.md`` (and .json)."""
    rows = comparison_rows(runs)
    cats = sorted({c for r in runs for c in r["summary"]["categories"]})
    lines = [f"# Model comparison — {runs[0]['run_id']}", "", f"Judge model: `{runs[0]['judge_model']}` · mode: {runs[0]['mode']}", "",
             "| Model | Pass rate | Passed | Mean judge score | p95 latency (ms) | App cost | Gate |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['model']}` | {row['pass_rate']:.1%} | {row['passed']} | {_fmt_score(row['mean_judge_score'])} | "
                     f"{row['p95_latency_ms']:.0f} | {_fmt_cost(row['total_cost_usd'], row['cost_estimated'])} | {'✅' if row['gate_passed'] else '❌'} |")
    lines += ["", "## Pass rate by category", "", "| Category | " + " | ".join(f"`{r['app_model']}`" for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for cat in cats:
        cells = [f"{r['summary']['categories'][cat]['pass_rate']:.0%}" if cat in r["summary"]["categories"] else "-" for r in runs]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-case results", "", "| Case | " + " | ".join(f"`{r['app_model']}`" for r in runs) + " |", "|---|" + "---|" * len(runs)]
    ids = [c["id"] for c in runs[0]["cases"]]
    lookup = [{c["id"]: c["passed"] for c in r["cases"]} for r in runs]
    for cid in ids:
        lines.append(f"| {cid} | " + " | ".join("✅" if lk.get(cid) else "❌" for lk in lookup) + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def print_model_comparison(runs: list[dict[str, Any]]) -> None:
    """Console side-by-side table."""
    t = Table(title="Model comparison")
    for col in ("Model", "Pass rate", "Passed", "Mean judge", "p95 ms", "App cost", "Gate"):
        t.add_column(col)
    for row in comparison_rows(runs):
        t.add_row(row["model"], f"{row['pass_rate']:.1%}", row["passed"], _fmt_score(row["mean_judge_score"]),
                  f"{row['p95_latency_ms']:.0f}", _fmt_cost(row["total_cost_usd"], row["cost_estimated"]),
                  "[green]PASS[/]" if row["gate_passed"] else "[red]FAIL[/]")
    console.print(t)
