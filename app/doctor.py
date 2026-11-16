"""Environment check: ``python -m app.doctor`` (or ``make doctor``).

1. Reports which env vars are set — NEVER their values (keys show only "set").
2. Makes one tiny live call to APP_MODEL and JUDGE_MODEL (and EXTRACTOR_MODEL/FALLBACK_MODEL if set)
   through LiteLLM and reports OK/FAIL, latency, and LiteLLM's view of tool-calling and
   response_format support for each model.
Use ``--offline`` to skip the live calls.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from app.llm_client import LiteLLMClient, LLMConfig, capability
from app.privacy import redact_secrets

console = Console()
SECRET_VARS = ("LITELLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "AZURE_API_KEY", "GEMINI_API_KEY")
MODEL_VARS = ("APP_MODEL", "JUDGE_MODEL", "EXTRACTOR_MODEL", "FALLBACK_MODEL")
OTHER_VARS = ("EVAL_MODE", "LITELLM_API_BASE", "AWS_ACCESS_KEY_ID", "AWS_REGION_NAME")


def env_report() -> list[tuple[str, str]]:
    """(name, status) for relevant env vars. Secrets and endpoints show only set/not set."""
    rows = []
    for name in OTHER_VARS + SECRET_VARS:
        rows.append((name, "set" if os.environ.get(name) else "-"))
    for name in MODEL_VARS:  # model names are not secret; showing them helps debugging
        rows.append((name, os.environ.get(name) or "-"))
    rows[0] = ("EVAL_MODE", os.environ.get("EVAL_MODE") or "(default: mock)")
    return rows


def _cap(v: bool | None) -> str:
    return "unknown" if v is None else ("yes" if v else "no")


def ping(client: LiteLLMClient, model: str) -> dict[str, str]:
    """One tiny completion; returns a status row (errors are redacted)."""
    row = {"model": model, "tools": _cap(capability("supports_function_calling", model)),
           "response_format": _cap(capability("supports_response_schema", model))}
    try:
        res = client.complete([{"role": "user", "content": "Reply with the single word OK."}], model=model, max_tokens=5, purpose="ping")
        row.update(status="OK", latency=f"{res.latency_ms:.0f} ms", detail=f"{res.total_tokens} tokens; reply={(res.content or '').strip()[:20]!r}")
    except Exception as exc:
        row.update(status="FAIL", latency="-", detail=redact_secrets(str(exc))[:200])
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(prog="python -m app.doctor")
    p.add_argument("--offline", action="store_true", help="only check env vars, no live calls")
    args = p.parse_args(argv)
    load_dotenv(override=False)

    t = Table(title="Environment (values of secrets are never shown)")
    t.add_column("Variable")
    t.add_column("Status")
    for name, status in env_report():
        t.add_row(name, status)
    console.print(t)

    cfg = LLMConfig.from_env(mode="mock", load_dotenv_file=False)  # mock: don't fail on missing models yet
    console.print(f"Connection style: [bold]{'LiteLLM proxy (api_base + api_key on every call)' if cfg.api_base else 'direct provider routing'}[/]")
    if cfg.api_base and not cfg.api_key:
        console.print("[yellow]warning:[/] LITELLM_API_BASE is set but LITELLM_API_KEY is empty.")
    app, judge = os.environ.get("APP_MODEL"), os.environ.get("JUDGE_MODEL")
    if app and judge and app == judge:
        console.print("[yellow]warning:[/] JUDGE_MODEL == APP_MODEL — self-preference bias; prefer a different judge.")
    if args.offline:
        return 0
    if not app or not judge:
        console.print("[red]APP_MODEL and JUDGE_MODEL must be set for live checks[/] (mock mode needs neither: `make eval`).")
        return 1

    client = LiteLLMClient(LLMConfig.from_env(mode="live", load_dotenv_file=False))
    models = [m for m in dict.fromkeys([app, judge, os.environ.get("EXTRACTOR_MODEL"), os.environ.get("FALLBACK_MODEL")]) if m]
    rt = Table(title="Live model checks via LiteLLM")
    for col in ("Model", "Status", "Latency", "Tools", "response_format", "Detail"):
        rt.add_column(col)
    rows = [ping(client, m) for m in models]
    for r in rows:
        color = "green" if r["status"] == "OK" else "red"
        rt.add_row(r["model"], f"[{color}]{r['status']}[/]", r["latency"], r["tools"], r["response_format"], r["detail"])
    console.print(rt)
    console.print("[dim]'unknown' = LiteLLM has no metadata for this model string (common for proxy aliases). "
                  "Override with TOOL_CALLING=native|prompt and STRUCTURED_OUTPUT=schema|prompt.[/]")
    return 0 if all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
