"""Sanity tests for the data layer and the retrieval metrics.

These are cheap checks that catch the mistakes that silently invalidate results:
wrong label vocabulary, a re-split dataset, test leakage into the clause bank,
and off-by-one rank arithmetic.

Run: pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.data import load_ledgar
from src.retrieval import ClauseBank, tokenize
from src.eval_retrieval import rank_of_first


@pytest.fixture(scope="module")
def dataset():
    return load_ledgar()


def test_official_splits(dataset):
    """The benchmark defines these sizes. Never re-split: comparisons to
    published LexGLUE numbers would break."""
    ds, _ = dataset
    assert len(ds["train"]) == 60_000
    assert len(ds["validation"]) == 10_000
    assert len(ds["test"]) == 10_000


def test_label_vocabulary(dataset):
    ds, names = dataset
    assert len(names) == 100
    assert len(set(names)) == 100, "duplicate label names would collapse classes"
    for split in ds:
        labels = np.asarray(ds[split]["label"])
        assert labels.min() >= 0
        assert labels.max() < len(names)


def test_no_leakage_between_train_and_test(dataset):
    """The clause bank is built from train, so train/test overlap is what the
    dedup scoring in evaluate.py exists to report. This pins the known figure
    so a silent data change cannot slip through."""
    ds, _ = dataset
    norm = lambda s: " ".join(s.lower().split())
    train_texts = {norm(t) for t in ds["train"]["text"]}
    overlap = sum(norm(t) in train_texts for t in ds["test"]["text"])
    assert 0 < overlap < 0.05 * len(ds["test"]), f"unexpected leakage: {overlap}"
    assert 200 <= overlap <= 400, f"overlap {overlap} differs from the documented 282"


def test_rank_of_first():
    assert rank_of_first(np.array([True, False])) == 1
    assert rank_of_first(np.array([False, False, True])) == 3
    assert rank_of_first(np.array([False, False])) is None


def test_tokenizer_keeps_numbers():
    """Queries cite '30-day notice' and '50% of shares'; alpha-only tokenisation
    would throw those away and make must_contain untestable."""
    assert "30" in tokenize("a 30-day notice period")
    assert "50" in tokenize("more than 50% of the voting shares")


def test_clause_bank_search_shapes():
    texts = ["the agreement shall be governed by new york law",
             "notice shall be given in writing",
             "the seller shall indemnify the buyer against losses"]
    bank = ClauseBank(texts=texts, labels=np.array([0, 1, 2]), names=["a", "b", "c"]).build_bm25()
    idx, scores = bank.search_bm25("governed by new york law", k=2)
    assert idx.shape == (2,) and scores.shape == (2,)
    assert idx[0] == 0, "the governing-law clause should rank first"
    assert scores[0] >= scores[1], "scores must come back sorted"


def test_queries_file_matches_label_vocabulary(dataset):
    """docs/queries.csv drives the retrieval evaluation; an unknown label there
    would crash the scoring with a KeyError far from the cause."""
    _, names = dataset
    q = pd.read_csv("docs/queries.csv")
    assert len(q) == 30
    unknown = [n for n in q.expected_label if n not in names]
    assert not unknown, f"unknown labels in queries.csv: {unknown}"
