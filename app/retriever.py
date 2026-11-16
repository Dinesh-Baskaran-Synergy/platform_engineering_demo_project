"""BM25 retrieval over the local markdown knowledge base (no vector DB).

Scores combine word-level BM25 with a down-weighted character-trigram BM25 so that
typos ("pasword resett") still find the right document.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.text_utils import content_terms, expand_terms

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


@dataclass(frozen=True)
class Document:
    """One knowledge-base document."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    """A retrieval hit."""

    doc_id: str
    title: str
    score: float
    text: str


def load_kb(kb_dir: Path = KB_DIR) -> list[Document]:
    """Load ``*.md`` docs; front matter supplies ``id`` and ``title``."""
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = raw
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2)
        docs.append(Document(meta.get("id", path.stem), meta.get("title", path.stem), body.strip()))
    return docs


def _trigrams(terms: list[str]) -> list[str]:
    grams = []
    for t in terms:
        padded = f"#{t}#"
        grams.extend(padded[i:i + 3] for i in range(len(padded) - 2))
    return grams


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lens = [len(doc) for doc in corpus]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        n = len(corpus)
        df = Counter(t for doc in corpus for t in set(doc))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tfs, self.lens):
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class Retriever:
    """BM25 retriever with a typo-tolerant trigram component."""

    def __init__(self, docs: list[Document] | None = None, trigram_weight: float = 0.3) -> None:
        self.docs = docs if docs is not None else load_kb()
        texts = [content_terms(d.title + " " + d.text) for d in self.docs]
        self._words = _BM25(texts)
        self._grams = _BM25([_trigrams(t) for t in texts])
        self.trigram_weight = trigram_weight

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.5) -> list[RetrievedDoc]:
        """Return up to ``k`` docs scoring at least ``min_score``, best first."""
        terms = expand_terms(content_terms(query))
        if not terms:
            return []
        w = self._words.scores(terms)
        g = self._grams.scores(_trigrams(terms))
        gmax = max(g) or 1.0
        wmax = max(w) or 1.0
        combined = [ws + self.trigram_weight * wmax * gs / gmax for ws, gs in zip(w, g)]
        ranked = sorted(zip(combined, self.docs), key=lambda x: (-x[0], x[1].doc_id))
        return [RetrievedDoc(d.doc_id, d.title, round(s, 4), d.text) for s, d in ranked[:k] if s >= min_score]

    def get(self, doc_id: str) -> Document | None:
        """Look up a document by id."""
        return next((d for d in self.docs if d.doc_id == doc_id), None)
