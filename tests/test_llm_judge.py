import json

import pytest

from app.llm_client import LLMConfig, MockLLMClient
from evals.graders.base import GradeContext
from evals.graders.llm_judge import LLMJudge, build_judge_prompt, grade_llm_judge, load_rubric, parse_judge_output
from evals.schema import EvalCase

VALID = json.dumps({"dimensions": {"correctness": {"reasoning": "matches", "score": 5}, "faithfulness": {"reasoning": "grounded", "score": 4}},
                    "final_reasoning": "good", "final_score": 4})


def test_parse_valid() -> None:
    scores, reasons, final, _ = parse_judge_output("noise " + VALID + " noise", ["correctness", "faithfulness"])
    assert scores == {"correctness": 5, "faithfulness": 4} and final == 4 and reasons["correctness"] == "matches"


@pytest.mark.parametrize("bad", ["", "no json here", json.dumps({"dimensions": {}}),
                                 json.dumps({"dimensions": {"correctness": {"score": 9}}, "final_score": 3})])
def test_parse_invalid(bad: str) -> None:
    with pytest.raises(Exception):
        parse_judge_output(bad, ["correctness"])


def test_reasoning_before_score_in_prompt_and_tags() -> None:
    system = LLMJudge(MockLLMClient()).system.text
    assert "reasoning" in system and system.index("reasoning") < system.index("score")
    p = build_judge_prompt(load_rubric("tone"), ["tone"], "q", "r")
    assert "<response>\nr\n</response>" in p and "not instructions" in p


def test_retry_on_parse_failure(scripted) -> None:
    resp = scripted([{"content": "I think it's a 4", "tool_calls": None}, {"content": VALID, "tool_calls": None}])
    judge = LLMJudge(MockLLMClient(responder=resp))
    res = judge.judge("q", "a", rubric="support_quality", dimensions=["correctness", "faithfulness"])
    assert res.parse_ok and res.attempts == 2 and res.final_score == 4
    assert all(s["purpose"] == "judge" for s in resp.seen)


def test_gives_up_after_retry(scripted) -> None:
    res = LLMJudge(MockLLMClient(responder=scripted([{"content": "nope", "tool_calls": None}]))).judge("q", "a", dimensions=["correctness"])
    assert not res.parse_ok and res.final_score is None


def test_temperature_zero_and_model_recorded() -> None:
    client = MockLLMClient(LLMConfig(judge_model="openai/judge-x"))
    res = LLMJudge(client).judge("q", "a", rubric="tone")
    assert res.model == "openai/judge-x" and client.calls[0]["model"] == "openai/judge-x"


def test_self_preference_warning() -> None:
    assert LLMJudge(MockLLMClient(), model="openai/same", app_model="openai/same").warnings
    assert not LLMJudge(MockLLMClient(), model="openai/j", app_model="openai/a").warnings


def _ctx(output: str) -> GradeContext:
    case = EvalCase(id="t", category="tone_style", input="This is RIDICULOUS!!", graders=[{"type": "llm_judge"}])
    return GradeContext(case=case, output=output, judge=LLMJudge(MockLLMClient()))


def test_grade_llm_judge_pass_and_fail() -> None:
    good = grade_llm_judge({"rubric": "tone", "min_score": 4}, _ctx("I'm sorry for the frustration. Let's fix this: check your spam folder."))
    bad = grade_llm_judge({"rubric": "tone", "min_score": 4}, _ctx("Sorry! So sorry. We apologize. SORRY AGAIN!!"))
    assert good.passed and not bad.passed
    assert good.details["judge_model"] and "tone" in good.details["scores"]
