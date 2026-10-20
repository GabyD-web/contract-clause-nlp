"""Clause-bank retrieval: BM25, sentence embeddings and a hybrid of the two.

The clause bank is built from the LEDGAR *train* split. The test split is never
indexed, so retrieval cannot return a clause the classifier was evaluated on.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation. Numbers are kept: queries cite 30-day
    notices and 50% thresholds, which a stopword/alpha-only filter would lose."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class ClauseBank:
    """Holds the clause texts, their labels and the two indexes."""
    texts: list[str]
    labels: np.ndarray
    names: list[str]
    _bm25: object = field(default=None, repr=False)
    _emb: np.ndarray = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def __len__(self):
        return len(self.texts)

    # ---------------------------------------------------------------- indexes
    def build_bm25(self):
        self._bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        return self

    def build_embeddings(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                         batch_size=64, device=None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        self._emb = self._model.encode(
            self.texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False).astype("float32")
        return self

    # ---------------------------------------------------------------- searches
    def search_bm25(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = self._bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

    def search_embeddings(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = self._model.encode([query], convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._emb @ q.astype("float32")          # cosine, vectors normalised
        idx = np.argsort(-scores)[:k]
        return idx, scores[idx]

    def search_hybrid(self, query: str, k: int = 10, k_each: int = 50,
                      method: str = "rrf", alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Fuse both rankings.

        rrf  : reciprocal rank fusion, scale-free, no score calibration needed.
        minmax: alpha * emb + (1 - alpha) * bm25 on min-max normalised scores.
        """
        b_idx, b_scores = self.search_bm25(query, k_each)
        e_idx, e_scores = self.search_embeddings(query, k_each)

        if method == "rrf":
            fused: dict[int, float] = {}
            for rank, i in enumerate(b_idx):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
            for rank, i in enumerate(e_idx):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
            idx = np.array(sorted(fused, key=lambda i: -fused[i])[:k])
            return idx, np.array([fused[int(i)] for i in idx])

        def minmax(s):
            lo, hi = float(np.min(s)), float(np.max(s))
            return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)

        scores = {}
        for i, s in zip(b_idx, minmax(b_scores)):
            scores[int(i)] = scores.get(int(i), 0.0) + (1 - alpha) * float(s)
        for i, s in zip(e_idx, minmax(e_scores)):
            scores[int(i)] = scores.get(int(i), 0.0) + alpha * float(s)
        idx = np.array(sorted(scores, key=lambda i: -scores[i])[:k])
        return idx, np.array([scores[int(i)] for i in idx])


def build_bank_from_dataset(ds, names, split="train", limit=None) -> ClauseBank:
    texts = list(ds[split]["text"])
    labels = np.asarray(ds[split]["label"])
    if limit:
        texts, labels = texts[:limit], labels[:limit]
    return ClauseBank(texts=texts, labels=labels, names=names)
