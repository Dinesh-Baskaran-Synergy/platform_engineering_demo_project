"""PII and secret redaction helpers.

Two different concerns live here:
* ``redact_pii`` masks customer PII (card numbers, passwords) in user input and
  in assistant output (defense in depth: the model is told not to echo secrets,
  and we also mask anything that slips through).
* ``redact_secrets`` scrubs API keys / Authorization headers from log lines and
  error messages so they never reach stdout, results files or CI logs.
"""
from __future__ import annotations

import os
import re

_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PASSWORD_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|passcode|contraseña)(\s*(?:is|:|=)\s*)(\S+)"
)
_CVV_RE = re.compile(r"(?i)\b(cvv|cvc|security code)(\s*(?:is|:|=)?\s*)(\d{3,4})\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_BEARER_RE = re.compile(r"(?i)(authorization\s*[:=]\s*)?bearer\s+[A-Za-z0-9._\-]+")
_KEY_RE = re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{8,}\b")


def _luhn_ok(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum (real card numbers do)."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_card(match: re.Match[str]) -> str:
    """Mask a card-number match, keeping only the last 4 digits."""
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return f"[CARD ****{digits[-4:]}]"
    return match.group(0)


def redact_pii(text: str) -> str:
    """Mask card numbers, CVVs and passwords in free text."""
    if not text:
        return text
    text = _CARD_RE.sub(mask_card, text)
    text = _CVV_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    text = _PASSWORD_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    text = _SSN_RE.sub("[SSN REDACTED]", text)
    text = _KEY_RE.sub("[REDACTED_KEY]", text)
    return text


def contains_card_number(text: str) -> bool:
    """True if ``text`` contains an unmasked, Luhn-valid card number."""
    for m in _CARD_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


_SECRET_ENV_VARS = (
    "LITELLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_API_KEY",
    "GEMINI_API_KEY",
)


def redact_secrets(text: str) -> str:
    """Remove API keys, bearer tokens and known secret env values from ``text``."""
    if not text:
        return text
    for var in _SECRET_ENV_VARS:
        val = os.environ.get(var)
        if val and len(val) >= 6:
            text = text.replace(val, "[REDACTED]")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _KEY_RE.sub("[REDACTED_KEY]", text)
    return text
