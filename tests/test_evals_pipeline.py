"""End-to-end harness tests in mock mode: datasets, runner, consistency, compare, calibration."""
from __future__ import annotations

from collections import Counter

import pytest

from app.llm_client import LLMConfig
from evals import compare, runner
from evals.graders.registry import GRADERS, run_grader
from evals.graders.base import GradeContext
from evals.judge_calibration.calibrate import agreement_stats, cohen_kappa, load_examples
from evals.schema import EvalCase, load_cases

CATEGORIES = {"happy_path", "edge_cases", "out_of_scope", "hallucination_faithfulness", "retrieval", "tool_use",
              "structured_extraction", "safety_refusals", "prompt_injection", "pii_privacy", "tone_style", "multi_turn",
              "format_compliance", "consistency", "regression", "performance", "model_portability"}


def test_datasets_cover_all_categories_with_known_graders() -> None:
    cases = load_cases()
    counts = Counter(c.category for c in cases)
    assert set(counts) == CATEGORIES
    assert all(n >= 3 for n in counts.values()) and 90 <= len(cases) <= 160
    for c in cases:
        for g in c.graders:
            assert g["type"] in GRADERS, f"{c.id}: unknown grader {g['type']}"
        assert c.severity in {"critical", "normal"}


def test_case_validation() -> None:
    with pytest.raises(Exception):
        EvalCase(id="x", category="c", graders=[{"type": "contains"}])  # no input
    with pytest.raises(Exception):
        EvalCase(id="x", category="c", input="hi", graders=[])


def test_unknown_or_crashing_grader_fails_safely() -> None:
    ctx = GradeContext(case=EvalCase(id="x", category="c", input="hi", graders=[{"type": "regex"}]))
    assert not run_grader({"type": "does_not_exist"}, ctx).passed
    assert "grader error" in run_grader({"type": "regex"}, ctx).reason  # missing 'pattern'


@pytest.fixture(scope="module")
def mock_run() -> dict:
    cases = load_cases(categories=["tool_use", "consistency", "retrieval", "structured_extraction", "performance"])
    return runner.run_suite(cases, LLMConfig(), runner.load_eval_config(), workers=4, progress=False)


def test_runner_records_everything(mock_run: dict) -> None:
    rec = next(c for c in mock_run["cases"] if c["id"] == "tu-001")
    for key in ("input", "output", "retrieved_ids", "tool_calls", "graders", "latency_ms", "prompt_tokens",
                "completion_tokens", "cost_usd", "prompt_version", "app_model", "judge_model"):
        assert key in rec
    assert rec["passed"] and rec["tool_calls"][0]["name"] == "get_order_status"
    assert mock_run["judge_model"] and mock_run["prompt_versions"]["assistant_system"].startswith("assistant_system@v1")


def test_runner_is_deterministic_in_mock_mode(mock_run: dict) -> None:
    cases = load_cases(categories=["tool_use"])
    again = runner.run_suite(cases, LLMConfig(), runner.load_eval_config(), workers=2, progress=False)
    first = {c["id"]: (c["passed"], c["output"]) for c in mock_run["cases"] if c["category"] == "tool_use"}
    assert first == {c["id"]: (c["passed"], c["output"]) for c in again["cases"]}


def test_consistency_group_grader_added(mock_run: dict) -> None:
    members = [c for c in mock_run["cases"] if c["category"] == "consistency"]
    assert members and all(any(g["grader"] == "consistency_group" for g in m["graders"]) for m in members)


def test_summary_gate_and_thresholds() -> None:
    cfg = {"thresholds": {"a": 0.5}, "default_threshold": 0.9, "overall_threshold": 0.5, "critical_must_pass": True}
    recs = [{"id": "a1", "category": "a", "passed": True, "severity": "normal", "latency_ms": 1, "score": 1},
            {"id": "a2", "category": "a", "passed": False, "severity": "critical", "latency_ms": 2, "score": 0}]
    s = runner.summarize(recs, cfg)
    assert s["categories"]["a"]["pass_rate"] == 0.5  # meets the 0.5 threshold...
    assert s["categories"]["a"]["critical_failures"] == ["a2"]
    assert not s["categories"]["a"]["met"] and not s["overall"]["gate_passed"]  # ...but a critical case failed
    cfg["critical_must_pass"] = False
    assert runner.summarize(recs, cfg)["overall"]["gate_passed"]


def test_p95() -> None:
    assert runner.p95([]) == 0.0 and runner.p95([1, 2, 3, 4, 100]) == 100 and runner.p95(list(range(1, 101))) == 95


def test_compare_flags_regressions(mock_run: dict) -> None:
    import copy

    current = copy.deepcopy(mock_run)
    current["cases"][0]["passed"] = False
    current["cases"][0]["graders"].append({"grader": "x", "passed": False, "reason": "broken", "score": 0})
    baseline = copy.deepcopy(mock_run)
    baseline["cases"][1]["passed"] = False
    d = compare.diff_runs(baseline, current)
    assert [r["id"] for r in d["regressions"]] == [current["cases"][0]["id"]]
    assert [i["id"] for i in d["improvements"]] == [baseline["cases"][1]["id"]]


def test_compare_cli_exit_codes(tmp_path, mock_run: dict) -> None:
    import copy
    import json

    base, cur = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps(mock_run, default=str))
    cur.write_text(json.dumps(mock_run, default=str))
    assert compare.main(["--baseline", str(base), "--current", str(cur)]) == 0
    broken = copy.deepcopy(mock_run)
    broken["cases"][0]["passed"] = False
    cur.write_text(json.dumps(broken, default=str))
    assert compare.main(["--baseline", str(base), "--current", str(cur)]) == 1
    assert compare.main(["--baseline", str(base), "--current", str(cur), "--allow-regressions"]) == 0


def test_calibration_set_and_stats() -> None:
    ex = load_examples()
    assert len(ex) == 20 and {e["human_pass"] for e in ex} == {True, False}
    assert cohen_kappa([True, False], [True, False]) == 1.0
    rows = [{"human_pass": True, "judge_pass": True, "human_score": 5, "judge_score": 4},
            {"human_pass": False, "judge_pass": True, "human_score": 1, "judge_score": 4}]
    s = agreement_stats(rows)
    assert s["pass_agreement"] == 0.5 and s["false_pass"] == 1 and s["score_mae"] == 2.0


def test_runner_cli_filters(tmp_path) -> None:
    code = runner.main(["--mode", "mock", "--ids", "hp-001,tu-001", "--out-dir", str(tmp_path), "--workers", "1"])
    assert code == 0 and (tmp_path / "latest.json").exists() and (tmp_path / "report.md").exists()


def test_multi_model_comparison(tmp_path) -> None:
    code = runner.main(["--mode", "mock", "--models", "openai/mock-a,openai/mock-b", "--out-dir", str(tmp_path), "--workers", "2"])
    assert code == 0
    md = (tmp_path / "model_comparison.md").read_text()
    assert "openai/mock-a" in md and "openai/mock-b" in md and "Pass rate" in md
