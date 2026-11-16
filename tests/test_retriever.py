import pytest

from app.retriever import Retriever, load_kb


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever()


def test_kb_loaded() -> None:
    docs = load_kb()
    assert 12 <= len(docs) <= 15
    assert len({d.doc_id for d in docs}) == len(docs)


@pytest.mark.parametrize("query,doc", [
    ("How much does the Pro plan cost?", "pricing"),
    ("refund for annual plan", "refunds"),
    ("HTTP 429 rate limit", "api-limits"),
    ("pasword resett", "password-reset"),  # typo tolerance via trigrams
    ("EU data residency", "regions"),
])
def test_top_doc(retriever: Retriever, query: str, doc: str) -> None:
    assert retriever.retrieve(query, k=3)[0].doc_id == doc


def test_empty_and_stopword_queries(retriever: Retriever) -> None:
    assert retriever.retrieve("") == []
    assert retriever.retrieve("the and of") == []


def test_k_respected_and_sorted(retriever: Retriever) -> None:
    hits = retriever.retrieve("plan pricing refund sla", k=2)
    assert len(hits) <= 2 and hits == sorted(hits, key=lambda h: -h.score)
