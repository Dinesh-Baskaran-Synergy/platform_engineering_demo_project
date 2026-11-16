"""Tiny text utilities shared by the retriever, the mock model and graders."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

STOPWORDS = frozenset(
    """a an the and or but if of to in on at for from by with about as is are was were be been being
    do does did doing have has had i me my we our you your it its this that these those there here
    what which who whom how when where why can could would should will shall may might must
    please tell know want need like just also any some all so than then too very not no yes
    hi hello hey thanks thank acmecloud acme get got im i'm i've id i'd let us much many
    per el la los las de del que mi mis es un una en y por para con como cómo cuánto cuanto puedo qué""".split()
)

_TOKEN_RE = re.compile(r"[^\W_][\w\-\.%$]*[^\W_]|[^\W_]|\$\d[\d.,]*")


def stem(word: str) -> str:
    """Very small suffix-stripping stemmer (good enough for a 14-doc KB)."""
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf == "es" and not w.endswith(("ses", "xes", "ches", "shes")):
                w = w[:-1]
            else:
                w = w[: -len(suf)]
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (keeps things like ord-1234, 99.9%, $29)."""
    return [t.lower().strip(".") for t in _TOKEN_RE.findall(text or "")]


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


# Query-side expansion for common support phrasings (stemmed forms).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "turn": ("enabl",), "activat": ("enabl",), "switch": ("enabl",),
    "money": ("refund",), "reimburs": ("refund",),
    "pric": ("cost",), "pay": ("payment",),
    "kept": ("retain",), "keep": ("retain",), "stor": ("retain",),
    "guarante": ("sla",), "commitment": ("sla",),
    "throttl": ("limit", "429"), "uptime": ("sla",),
    "2fa": ("two-factor",), "two-factor": ("2fa",),
    "wait": ("receiv",), "arriv": ("receiv",), "never": ("receiv",),
}


def expand_terms(terms: list[str]) -> list[str]:
    """Add synonyms for query terms (order preserved, no duplicates)."""
    out = list(dict.fromkeys(terms))
    for t in terms:
        for syn in SYNONYMS.get(t, ()):
            if syn not in out:
                out.append(syn)
    return out


def fuzzy_in(term: str, vocab: set[str], threshold: float = 0.8) -> bool:
    """True if ``term`` is in ``vocab`` or (for longer words) close to a vocab word — handles typos."""
    if term in vocab:
        return True
    if len(term) < 5:
        return False
    return any(
        abs(len(v) - len(term)) <= 2 and SequenceMatcher(None, term, v).ratio() >= threshold
        for v in vocab
    )


def split_sentences(text: str) -> list[str]:
    """Split markdown body text into sentences (one fact per line in our KB)."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s.strip())
    return out


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len((text or "").split())


_FOLLOWUP_RE = re.compile(
    r"^(and|what about|how about|also|ok(ay)?,? (and|what|how)|same for|is that|does that|what if|actually)\b"
    r"|\b(it|that|this|those|them|instead)\b",
    re.IGNORECASE,
)


def is_followup(text: str) -> bool:
    """Heuristic: does this user turn depend on the previous one (short or anaphoric)?"""
    return len(content_terms(text)) <= 4 or bool(_FOLLOWUP_RE.search(text or ""))
