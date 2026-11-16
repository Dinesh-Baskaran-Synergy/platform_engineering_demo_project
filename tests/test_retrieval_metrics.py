from evals.graders.retrieval_metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["a", "z"], 2) == 0.5
    assert recall_at_k([], [], 3) == 1.0


def test_mrr() -> None:
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0
    assert mean_reciprocal_rank([(["a"], ["a"]), (["x", "a"], ["a"])]) == 0.75
    assert mean_reciprocal_rank([]) == 0.0
