"""Evaluate clause-bank retrieval on the 30 lawyer queries.

Metrics, all at k = 5 (the model card's protocol):
  hit@k          the top-k contains at least one clause with the expected label
  mrr            mean reciprocal rank of the first correct-label hit
  must_contain   fraction of queries whose must_contain string occurs in the
                 top-k text (case-insensitive); only queries that define one count

BM25 only, embeddings only, and hybrid are compared. A hybrid's fusion method
and weight are chosen on the *head/torso* half of the queries only; the rare-class
half is reported untouched, so the headline number is not tuned on itself.
"""
import argparse, json, os
import numpy as np
import pandas as pd

from src.data import load_ledgar
from src.retrieval import build_bank_from_dataset, tokenize
from src.utils import save_json


def rank_of_first(correct: np.ndarray) -> int | None:
    hits = np.flatnonzero(correct)
    return int(hits[0]) + 1 if len(hits) else None


def evaluate(searchers, queries: pd.DataFrame, names, labels: np.ndarray,
             texts: list[str], k=5):
    name2id = {n: i for i, n in enumerate(names)}
    expected = np.array([name2id[q] for q in queries.expected_label])
    rows = []
    for method, search in searchers.items():
        hits, rrs, must_hits, must_n = 0, [], 0, 0
        ranks = []
        for qi, q in enumerate(queries.itertuples()):
            idx, _ = search(q.query, k)
            correct = labels[idx] == expected[qi]
            r = rank_of_first(correct)
            ranks.append(r if r is not None else np.inf)
            hits += r is not None
            rrs.append(1.0 / r if r is not None else 0.0)
            if isinstance(q.must_contain, str) and q.must_contain.strip():
                must_n += 1
                needle = q.must_contain.strip().lower()
                top = [texts[i].lower() for i in idx]
                must_hits += any(needle in t for t in top)
        rows.append({
            "method": method,
            f"hit@{k}": hits / len(queries),
            "mrr": float(np.mean(rrs)),
            "must_contain_topk": (must_hits / must_n) if must_n else None,
            "n_must_contain": must_n,
        })
    return pd.DataFrame(rows)


def main(bank_split="train", limit=None, k=5, out="results/retrieval"):
    os.makedirs(out, exist_ok=True)
    ds, names = load_ledgar()
    queries = pd.read_csv("docs/queries.csv")
    bank = build_bank_from_dataset(ds, names, bank_split, limit)
    bank.build_bm25().build_embeddings()
    print(f"clause bank: {len(bank)} clauses from the {bank_split} split")

    searchers = {
        "BM25": bank.search_bm25,
        "Embeddings": bank.search_embeddings,
    }
    base = evaluate(searchers, queries, names, bank.labels, bank.texts, k=k)

    # --- tune the hybrid on head/torso queries only -------------------------
    train_n = np.bincount(np.asarray(ds["train"]["label"]), minlength=len(names))
    name2id = {n: i for i, n in enumerate(names)}
    q_freq = np.array([train_n[name2id[q]] for q in queries.expected_label])
    # per the EDA bucketing: head = 20 most frequent, torso = next 40, tail = rest
    order = np.argsort(-np.bincount(np.asarray(ds["train"]["label"]), minlength=len(names)))
    head_ids, torso_ids = set(order[:20]), set(order[20:60])
    is_tail = np.array([name2id[q] not in head_ids | torso_ids for q in queries.expected_label])
    tune = queries[~is_tail].reset_index(drop=True)
    rare = queries[is_tail].reset_index(drop=True)

    best, best_cfg = -1.0, None
    for method, alpha in [("rrf", None)] + [("minmax", a) for a in (0.3, 0.5, 0.7)]:
        s = {"Hybrid": lambda q, kk, m=method, a=alpha: bank.search_hybrid(q, kk, method=m, alpha=a or 0.5)}
        r = evaluate(s, tune, names, bank.labels, bank.texts, k=k).iloc[0]
        print(f"  hybrid {method} alpha={alpha}: tune hit@{k}={r[f'hit@{k}']:.3f} mrr={r['mrr']:.3f}")
        if r["mrr"] > best:
            best, best_cfg = r["mrr"], (method, alpha)

    method, alpha = best_cfg
    searchers["Hybrid"] = lambda q, kk: bank.search_hybrid(q, kk, method=method, alpha=alpha or 0.5)
    full = evaluate(searchers, queries, names, bank.labels, bank.texts, k=k)

    # performance split by bucket, to show where retrieval struggles
    per_bucket = []
    for label, subset in [("head+torso", tune), ("tail", rare)]:
        r = evaluate(searchers, subset, names, bank.labels, bank.texts, k=k)
        r.insert(1, "bucket", label); r.insert(2, "n_queries", len(subset))
        per_bucket.append(r)
    per_bucket = pd.concat(per_bucket, ignore_index=True)

    full.to_csv(f"{out}/summary.csv", index=False)
    per_bucket.to_csv(f"{out}/by_bucket.csv", index=False)
    save_json({"bank_split": bank_split, "bank_size": len(bank), "k": k,
               "hybrid": {"method": method, "alpha": alpha},
               "n_queries": len(queries)}, f"{out}/config.json")
    print("\n", full.to_string(index=False))
    print("\n", per_bucket.to_string(index=False))
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/retrieval")
    a = ap.parse_args()
    main(a.bank_split, a.limit, a.k, a.out)
