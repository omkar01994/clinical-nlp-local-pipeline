import pytest

from clinical_pipeline.evaluation import Item, auroc, evaluate, first_hit_rank, ranking_metrics
from tests.test_pipeline import make_pipeline


def test_first_hit_rank():
    assert first_hit_rank(["a", "b", "c"], ["b", "c"]) == 2
    assert first_hit_rank(["a", "b"], ["z"]) is None


def test_ranking_metrics():
    m = ranking_metrics([1, 2, None, 4])
    assert m["top1"] == 0.25
    assert m["recall@3"] == 0.5
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 4)


def test_auroc():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1], [1]) == 0.5


def test_evaluate_end_to_end_with_fakes():
    retriever = make_pipeline().retriever
    items = [
        Item("blood pressure salt", ["bp"], "keyword"),
        Item("fever temperature", ["fever"], "keyword"),
        Item("asthma inhaler", ["asthma"], "paraphrase"),
        Item("pizza", [], "non-medical"),
    ]
    r = evaluate(items, retriever, alphas=[0.0, 1.0])
    assert r["ranking"]["BM25 only"]["top1"] == 1.0
    assert r["end_to_end"]["overall"] == 1.0
    assert r["gate_auroc"]["BM25 max score"] == 1.0
    assert r["failures"] == []
