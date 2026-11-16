import pytest

from app import doctor, llm_client
from app.llm_client import LiteLLMClient, LLMConfig
from app.prompt_loader import load_prompt


def test_env_report_never_shows_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "sk-should-never-print-1234")
    monkeypatch.setenv("LITELLM_API_BASE", "https://internal-proxy.example")
    monkeypatch.setenv("APP_MODEL", "openai/app-alias")
    rows = dict(doctor.env_report())
    assert rows["LITELLM_API_KEY"] == "set" and rows["LITELLM_API_BASE"] == "set"
    assert rows["APP_MODEL"] == "openai/app-alias"
    assert "sk-should-never-print-1234" not in str(rows)


def test_ping_ok_and_fail_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    real = llm_client.litellm.completion
    monkeypatch.setattr(llm_client.litellm, "completion",
                        lambda **kw: real(model=kw["model"], messages=kw["messages"], mock_response="OK"))
    client = LiteLLMClient(LLMConfig(mode="live", app_model="openai/a", judge_model="openai/j"))
    assert doctor.ping(client, "openai/gpt-4o-mini")["status"] == "OK"

    monkeypatch.setenv("LITELLM_API_KEY", "sk-secret-in-error-777")

    def boom(**kw):
        raise RuntimeError("auth failed for key sk-secret-in-error-777")

    monkeypatch.setattr(llm_client.litellm, "completion", boom)
    row = doctor.ping(client, "openai/unknown-alias")
    assert row["status"] == "FAIL" and "sk-secret-in-error-777" not in row["detail"] and row["tools"] == "unknown"


def test_doctor_offline_exit_code() -> None:
    assert doctor.main(["--offline"]) == 0


def test_prompt_versions_are_traceable() -> None:
    p = load_prompt("assistant_system")
    assert p.version.startswith("assistant_system@v1#") and len(p.version.split("#")[1]) == 8
    with pytest.raises(FileNotFoundError):
        load_prompt("assistant_system", version=99)
