"""Eval runner CLI.

Examples:
    python -m evals.runner                                   # all categories, EVAL_MODE from env (default mock)
    python -m evals.runner --category tool_use --mode mock --workers 4
    python -m evals.runner --mode live --app-model openai/my-alias --judge-model openai/judge-alias
    python -m evals.runner --mode live --models openai/a,anthropic/b   # side-by-side comparison

Exit code is non-zero when the pass criteria in ``evals/config.yaml`` are not met
(use ``--no-fail`` to always exit 0).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.assistant import Assistant
from app.extractor import Extractor
from app.llm_client import ConfigError, LLMClient, LLMConfig, LLMResult, get_client
from app.privacy import redact_secrets
from app.prompt_loader import load_prompt
from app.retriever import Retriever
from app.text_utils import content_terms
from evals.graders.base import GradeContext, GradeResult
from evals.graders.llm_judge import LLMJudge
from evals.graders.registry import run_grader
from evals.schema import EvalCase, load_cases

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).parent / "config.yaml"
log = logging.getLogger("acme.runner")


def load_eval_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``evals/config.yaml``."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ metrics
def _sum_calls(calls: list[LLMResult], pricing: dict[str, float]) -> dict[str, Any]:
    """Aggregate tokens/cost over LLM calls. Unpriced calls get a token-based estimate."""
    pt = sum(c.prompt_tokens for c in calls)
    ct = sum(c.completion_tokens for c in calls)
    cost, estimated = 0.0, False
    for c in calls:
        if c.cost_usd is not None:
            cost += c.cost_usd
        else:
            estimated = True
            cost += c.prompt_tokens / 1000 * pricing["prompt"] + c.completion_tokens / 1000 * pricing["completion"]
    return {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": round(cost, 8) if calls else 0.0,
            "cost_estimated": estimated, "llm_calls": len(calls)}


def p95(values: list[float]) -> float:
    """95th percentile (nearest-rank)."""
    if not values:
        return 0.0
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


# ------------------------------------------------------------------ one case
def run_case(case: EvalCase, client: LLMClient, judge: LLMJudge, retriever: Retriever,
             eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the system under test for one case, then run all its graders."""
    pricing = eval_cfg["fallback_pricing_usd_per_1k_tokens"]
    k = int(eval_cfg.get("retrieval", {}).get("k", 3))
    rec: dict[str, Any] = {
        "id": case.id, "category": case.category, "task": case.task, "severity": case.severity, "tags": case.tags,
        "group": case.group, "input": case.question_text(), "output": "", "retrieved_ids": [], "tool_calls": [],
        "extraction": None, "graders": [], "passed": False, "error": None, "warnings": [],
        "app_model": client.config.app_model if case.task != "extract" else client.config.extractor_model,
        "judge_model": judge.model, "prompt_version": None,
    }
    ctx = GradeContext(case=case, judge=judge)
    start = time.perf_counter()
    try:
        if case.task == "retrieve":
            docs = retriever.retrieve(case.input or "", k=k)
            ctx.retrieved_ids = [d.doc_id for d in docs]
            ctx.output = ", ".join(ctx.retrieved_ids)
            calls: list[LLMResult] = []
            rec["app_model"] = "bm25 (no LLM)"
        elif case.task == "extract":
            ex = Extractor(client)
            res = ex.extract(case.input or "")
            ctx.extraction, ctx.extraction_raw = res.data, res.raw
            ctx.output = json.dumps(res.data) if res.data else (res.raw or "")
            calls = res.calls
            rec["prompt_version"] = ex.prompt.version
            rec["extraction"] = {"data": res.data, "valid": res.valid, "attempts": res.attempts,
                                 "used_response_schema": res.used_response_schema, "error": res.error}
        else:
            assistant = Assistant(client, retriever=retriever, k=k)  # fresh MockDB per case
            resp = assistant.respond(case.conversation())
            ctx.output = resp.answer
            ctx.retrieved_ids = resp.retrieved_ids
            ctx.tool_calls = [{"name": t.name, "arguments": t.arguments, "result": t.result} for t in resp.tool_calls]
            tool_ctx = "\n".join(f"[tool {t.name}] {json.dumps(t.result)}" for t in resp.tool_calls)
            ctx.retrieved_text = "\n\n".join(f"[{d.doc_id}]\n{d.text}" for d in resp.retrieved) + ("\n\n" + tool_ctx if tool_ctx else "")
            calls = resp.calls
            rec["prompt_version"] = resp.prompt_version
            rec["warnings"] = resp.warnings
            rec["tool_mode"] = resp.tool_mode
    except Exception as exc:  # system-under-test crash = case failure, not a runner crash
        rec["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")
        rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        rec.update(_sum_calls([], pricing))
        return rec
    ctx.latency_ms = (time.perf_counter() - start) * 1000
    m = _sum_calls(calls, pricing)
    ctx.prompt_tokens, ctx.completion_tokens = m["prompt_tokens"], m["completion_tokens"]
    ctx.cost_usd, ctx.cost_estimated = m["cost_usd"], m["cost_estimated"]

    judge_calls: list[LLMResult] = []
    for spec in case.graders:
        g = run_grader(spec, ctx)
        judge_calls += g.details.pop("calls", [])
        rec["graders"].append(asdict(g))
    jm = _sum_calls(judge_calls, pricing)
    rec.update(
        output=ctx.output, retrieved_ids=ctx.retrieved_ids, tool_calls=ctx.tool_calls,
        latency_ms=round(ctx.latency_ms, 2), **m,
        judge_tokens=jm["prompt_tokens"] + jm["completion_tokens"], judge_cost_usd=jm["cost_usd"],
        passed=all(g["passed"] for g in rec["graders"]),
    )
    scores = [g["details"]["final_score"] for g in rec["graders"] if g["grader"].startswith("llm_judge") and g["details"].get("final_score")]
    rec["judge_score"] = round(sum(scores) / len(scores), 2) if scores else None
    rec["score"] = round(sum(g["score"] for g in rec["graders"]) / len(rec["graders"]), 4)
    return rec


# ------------------------------------------------------------ consistency
def _jaccard(a: str, b: str) -> float:
    sa, sb = set(content_terms(a)), set(content_terms(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def apply_consistency(records: list[dict[str, Any]], cases: dict[str, EvalCase], min_similarity: float) -> None:
    """Group-level check: paraphrases must share key facts and be lexically similar."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("group"):
            groups.setdefault(r["group"], []).append(r)
    for gname, members in groups.items():
        facts = cases[members[0]["id"]].expected.get("key_facts", [])
        missing = {m["id"]: [f for f in facts if f.lower() not in m["output"].lower()] for m in members}
        sims = [_jaccard(a["output"], b["output"]) for i, a in enumerate(members) for b in members[i + 1:]]
        min_sim = min(sims) if sims else 1.0
        ok = not any(missing.values()) and min_sim >= min_similarity
        reason = f"group '{gname}' ({len(members)} paraphrases): min similarity {min_sim:.2f} (need {min_similarity})"
        if any(missing.values()):
            reason += f"; missing key facts {{{', '.join(f'{k}: {v}' for k, v in missing.items() if v)}}}"
        for m in members:
            m["graders"].append(asdict(GradeResult("consistency_group", ok, min_sim, reason)))
            m["passed"] = m["passed"] and ok


# ------------------------------------------------------------ suite
def summarize(records: list[dict[str, Any]], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Per-category and overall stats plus the pass/fail gate."""
    cats: dict[str, dict[str, Any]] = {}
    for cat in sorted({r["category"] for r in records}):
        rs = [r for r in records if r["category"] == cat]
        passed = sum(r["passed"] for r in rs)
        judge_scores = [r["judge_score"] for r in rs if r.get("judge_score")]
        thr = float(eval_cfg.get("thresholds", {}).get(cat, eval_cfg.get("default_threshold", 0.9)))
        crit_fail = [r["id"] for r in rs if r["severity"] == "critical" and not r["passed"]]
        rate = passed / len(rs)
        cats[cat] = {
            "cases": len(rs), "passed": passed, "pass_rate": round(rate, 4),
            "mean_score": round(sum(r.get("score", 0) for r in rs) / len(rs), 4),
            "mean_judge_score": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
            "p95_latency_ms": round(p95([r["latency_ms"] for r in rs]), 1),
            "cost_usd": round(sum(r.get("cost_usd", 0) + r.get("judge_cost_usd", 0) for r in rs), 6),
            "threshold": thr, "critical_failures": crit_fail,
            "met": rate >= thr and not (eval_cfg.get("critical_must_pass", True) and crit_fail),
        }
    n = len(records)
    overall_rate = sum(r["passed"] for r in records) / n if n else 0.0
    judge_all = [r["judge_score"] for r in records if r.get("judge_score")]
    overall = {
        "cases": n, "passed": sum(r["passed"] for r in records), "pass_rate": round(overall_rate, 4),
        "mean_judge_score": round(sum(judge_all) / len(judge_all), 2) if judge_all else None,
        "p95_latency_ms": round(p95([r["latency_ms"] for r in records]), 1),
        "app_cost_usd": round(sum(r.get("cost_usd", 0) for r in records), 6),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in records), 6),
        "cost_estimated": any(r.get("cost_estimated") for r in records),
        "threshold": float(eval_cfg.get("overall_threshold", 0.85)),
    }
    overall["total_cost_usd"] = round(overall["app_cost_usd"] + overall["judge_cost_usd"], 6)
    failed_cats = [c for c, s in cats.items() if not s["met"]]
    overall["gate_passed"] = overall_rate >= overall["threshold"] and not failed_cats
    overall["failed_categories"] = failed_cats
    return {"categories": cats, "overall": overall}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def run_suite(cases: list[EvalCase], config: LLMConfig, eval_cfg: dict[str, Any], workers: int = 4,
              client: LLMClient | None = None, progress: bool = True) -> dict[str, Any]:
    """Run all ``cases`` against one app model; returns the full run record."""
    client = client or get_client(config)
    judge = LLMJudge(client, model=config.judge_model, app_model=config.app_model)
    retriever = Retriever()
    by_id = {c.id: c for c in cases}
    started = datetime.now(timezone.utc)

    def _one(case: EvalCase) -> dict[str, Any]:
        return run_case(case, client, judge, retriever, eval_cfg)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        if progress:
            from rich.progress import Progress

            with Progress(transient=True) as bar:
                task = bar.add_task(f"[cyan]{config.app_model}", total=len(cases))
                for rec in pool.map(_one, cases):
                    records.append(rec)
                    bar.advance(task)
        else:
            records = list(pool.map(_one, cases))
    apply_consistency(records, by_id, float(eval_cfg.get("consistency", {}).get("min_similarity", 0.5)))
    records.sort(key=lambda r: (r["category"], r["id"]))
    import litellm

    return {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started.isoformat(),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
        "mode": config.mode,
        "app_model": config.app_model,
        "judge_model": config.judge_model,
        "extractor_model": config.extractor_model,
        "connection": "proxy" if config.api_base else "direct",
        "litellm_version": getattr(litellm, "__version__", None) or _pkg_version("litellm"),
        "git_commit": _git_commit(),
        "prompt_versions": {n: load_prompt(n).version for n in ("assistant_system", "extractor_system", "judge_system")},
        "warnings": judge.warnings,
        "summary": summarize(records, eval_cfg),
        "cases": records,
    }


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")[:60]


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.runner", description="Run AcmeCloud evals.")
    p.add_argument("--category", help="comma-separated categories (default: all; 'all' = all)")
    p.add_argument("--ids", help="comma-separated case ids")
    p.add_argument("--tags", help="comma-separated tags (case must have at least one)")
    p.add_argument("--mode", choices=["mock", "live"], help="overrides EVAL_MODE")
    p.add_argument("--workers", type=int, default=4, help="parallel cases (lower it if you hit rate limits)")
    p.add_argument("--app-model", help="LiteLLM model string for the assistant (overrides APP_MODEL)")
    p.add_argument("--judge-model", help="LiteLLM model string for the judge (overrides JUDGE_MODEL)")
    p.add_argument("--models", help="comma-separated app models to compare (default category: model_portability)")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--no-fail", action="store_true", help="always exit 0")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def select_cases(args: argparse.Namespace) -> list[EvalCase]:
    """Apply --category/--ids/--tags filters."""
    cats = None
    if args.category and args.category != "all":
        cats = [c.strip() for c in args.category.split(",")]
    elif args.models and not args.category:
        cats = ["model_portability"]
    cases = load_cases(categories=cats)
    if args.ids:
        ids = {i.strip() for i in args.ids.split(",")}
        cases = [c for c in cases if c.id in ids]
    if args.tags:
        tags = {t.strip() for t in args.tags.split(",")}
        cases = [c for c in cases if tags & set(c.tags)]
    return cases


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    from evals import report

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    eval_cfg = load_eval_config()
    cases = select_cases(args)
    if not cases:
        print("No cases matched the filters.", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [None]
    runs = []
    try:
        for model in models:
            config = LLMConfig.from_env(mode=args.mode, app_model=model or args.app_model, judge_model=args.judge_model)
            run = run_suite(cases, config, eval_cfg, workers=args.workers)
            runs.append(run)
            suffix = f"_{_slug(config.app_model)}" if model else ""
            report.write_run(run, out_dir, suffix=suffix, latest=len(models) == 1)
            report.print_run(run)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if len(runs) > 1:
        path = report.write_model_comparison(runs, out_dir)
        report.print_model_comparison(runs)
        print(f"Model comparison written to {path}")
    gate = all(r["summary"]["overall"]["gate_passed"] for r in runs)
    return 0 if gate or args.no_fail else 1


if __name__ == "__main__":
    sys.exit(main())
