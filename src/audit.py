"""Bias / robustness audit of a trained clause classifier.

Produces results/audit/summary.csv, the table the model card asks for:
  * frequent vs rare clause types
  * clauses longer than max_length (the truncation question the EDA raised
    but never answered)
  * French clauses and non-clause text (abstention behaviour)
  * a confidence-threshold sweep, for the "X% tagged automatically" claim
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_ledgar
from src.predict import ClauseClassifier
from src.utils import save_json


def confidence_stats(probs: np.ndarray) -> dict:
    return {"n": int(len(probs)),
            "mean_confidence": float(probs.max(1).mean()),
            "median_confidence": float(np.median(probs.max(1))),
            "pct_conf_gt_0.9": float((probs.max(1) > 0.9).mean())}


def main(model_dir, out_dir, max_length=256):
    os.makedirs(out_dir, exist_ok=True)
    ds, names = load_ledgar()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    clf = ClauseClassifier(model_dir, max_length=max_length)

    test = ds["test"].to_pandas()
    train = ds["train"].to_pandas()
    y = test.label.values
    train_n = np.bincount(train.label.values, minlength=len(names))

    # ---- classifier predictions on the full test set -----------------------
    probs = clf.predict_proba(test.text.tolist())
    pred = probs.argmax(1)
    np.save(f"{out_dir}/test_probs.npy", probs)
    overall_acc = accuracy_score(y, pred)
    overall_macro = f1_score(y, pred, average="macro")
    # Per-class F1 on the FULL test set. Slicing rows and recomputing macro-F1
    # inside a slice would average over absent classes (scored 0) and be
    # meaningless, so slices reuse these global per-class scores instead.
    per_class_f1 = f1_score(y, pred, average=None, labels=list(range(len(names))),
                            zero_division=0)

    # clause length, measured with the real tokenizer (not a proxy)
    lens = np.array([len(tok(t)["input_ids"]) for t in test.text.tolist()])
    test["n_tokens"] = lens
    test["pred"] = pred
    test["correct"] = pred == y
    test["confidence"] = probs.max(1)
    np.save(f"{out_dir}/test_tokens.npy", lens)

    rows = []

    def add(slice_name, mask, note=""):
        sub = test[mask]
        if len(sub) == 0:
            return
        present = np.unique(sub.label.values)   # classes that occur in this slice
        rows.append({
            "slice": slice_name, "n": int(len(sub)),
            "n_classes": int(len(present)),
            "accuracy": accuracy_score(sub.label, sub.pred),
            "macro_f1_over_present_classes": float(per_class_f1[present].mean()),
            "mean_confidence": float(sub.confidence.mean()),
            "gap_vs_overall": accuracy_score(sub.label, sub.pred) - overall_acc,
            "note": note,
        })

    # ---- frequency slices --------------------------------------------------
    order = np.argsort(-train_n)
    head, torso = set(order[:20]), set(order[20:60])
    rare_ids = set(order[60:])
    add("all test clauses", np.ones(len(test), bool), "reference row")
    add("frequent clause types (top 20)", test.label.isin(head).values)
    add("mid-frequency clause types", test.label.isin(torso).values)
    add("rare clause types (<50 train examples)",
        (train_n[test.label.values] < 50), "unreliable by construction")
    add("rarest bucket (bottom 40 by train count)", test.label.isin(rare_ids).values)

    # ---- the truncation question ------------------------------------------
    long_mask = lens > max_length
    add(f"clauses > {max_length} tokens (truncated)", long_mask,
        "input is cut at max_length")
    add(f"clauses <= {max_length} tokens", ~long_mask)
    for r in rows:
        if r["slice"].startswith("clauses >"):
            short_acc = next(x["accuracy"] for x in rows if x["slice"].startswith("clauses <="))
            r["note"] += f"; accuracy gap vs short clauses = {r['accuracy'] - short_acc:+.3f}"

    summary = pd.DataFrame(rows)
    summary.to_csv(f"{out_dir}/summary.csv", index=False)

    # ---- out-of-scope and French text --------------------------------------
    oos_path = "docs/out_of_scope.txt"
    oos = {}
    if os.path.exists(oos_path):
        with open(oos_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # the file lists English news, chat, legal-not-clause, then French lines
        fr = [l for l in lines if _is_french(l)]
        en = [l for l in lines if l not in fr]
        for label, group in [("non-clause English text", en), ("French text", fr)]:
            if not group:
                continue
            p = clf.predict_proba(group)
            oos[label] = {"n": len(group), **confidence_stats(p),
                          "predicted_labels": pd.Series(p.argmax(1))
                          .map(lambda i: names[i]).value_counts().head(5).to_dict()}
    save_json(oos, f"{out_dir}/out_of_scope.json")

    # ---- abstention sweep --------------------------------------------------
    sweep = []
    for t in np.arange(0.5, 0.99, 0.05):
        keep = test.confidence.values >= t
        if keep.sum() < 50:
            break
        sweep.append({"threshold": round(float(t), 2),
                      "coverage": float(keep.mean()),
                      "accuracy_on_kept": accuracy_score(y[keep], pred[keep]),
                      "accuracy_on_referred": accuracy_score(y[~keep], pred[~keep])
                      if (~keep).sum() else None})
    pd.DataFrame(sweep).to_csv(f"{out_dir}/abstention.csv", index=False)

    save_json({"model_dir": model_dir, "max_length": max_length,
               "overall_accuracy": float(overall_acc),
               "overall_macro_f1": float(overall_macro),
               "n_truncated": int(long_mask.sum()),
               "pct_truncated": float(long_mask.mean())}, f"{out_dir}/meta.json")

    print(summary.to_string(index=False))
    print("\nout-of-scope:", json.dumps(oos, indent=2)[:800])
    print("\nabstention:\n", pd.DataFrame(sweep).to_string(index=False))


def _is_french(text: str) -> bool:
    """Cheap heuristic: French function words are absent from the English rows
    in this file, and the accented characters are a strong signal."""
    markers = (" le ", " la ", " les ", " est ", " sera ", " par ", " pour ",
               "s'engage", "présent", "réuni", "prix ", "merci", "conseil",
               "chaque", "toute", "être")
    t = f" {text.lower()} "
    return sum(m in t for m in markers) >= 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="runs/legalbert_e10_s42")
    ap.add_argument("--out_dir", default="results/audit")
    ap.add_argument("--max_length", type=int, default=256)
    a = ap.parse_args()
    main(a.model_dir, a.out_dir, a.max_length)
