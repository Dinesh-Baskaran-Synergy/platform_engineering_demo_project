from app.privacy import contains_card_number, redact_pii, redact_secrets


def test_card_masked_keeps_last4() -> None:
    out = redact_pii("my card 4111 1111 1111 1111 thanks")
    assert "4111 1111 1111 1111" not in out and "[CARD ****1111]" in out
    assert not contains_card_number(out)


def test_non_luhn_number_untouched() -> None:
    assert redact_pii("order 1234 5678 9012 3456") == "order 1234 5678 9012 3456"


def test_password_cvv_ssn_keys() -> None:
    out = redact_pii("password: Hunter2! cvv 123 ssn 123-45-6789 key sk-live-abcdef1234567890")
    for secret in ("Hunter2!", "123-45-6789", "sk-live-abcdef1234567890"):
        assert secret not in out
    assert "cvv [redacted]" in out.lower()


def test_redact_secrets_env_and_bearer(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "super-secret-key-1")
    out = redact_secrets("key=super-secret-key-1 Authorization: Bearer abc.def-123")
    assert "super-secret-key-1" not in out and "abc.def-123" not in out
