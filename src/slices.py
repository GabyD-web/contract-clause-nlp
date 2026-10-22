"""Audit by kind of contract and party role, from predictions already saved (no GPU, no model).

Each test clause is tagged with the contract families whose keywords it contains (a clause can be
in several, or in none). For each family we report accuracy and macro-F1 of TF-IDF, Legal-BERT
(each saved seed, and their mean) and the ensemble, plus Legal-BERT's mean confidence.

Usage (repo root):  python -m src.slices
Outputs:            results/audit/slices.csv, results/audit/slices.json
Caveat: families are inferred from keywords, so they also differ in which clause types they contain;
macro-F1 is computed over the types present in each slice.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_ledgar

OUT_DIR = "results/audit"
SEEDS = {42: None, 1: 1.45, 2: 1.43}   # seed -> temperature fitted on validation (None: not fitted, use 1.44)

FAMILIES = {
    "lease / real estate (landlord, tenant)": r"\b(landlord|tenant|lessor|lessee|leased premises|sublease)\b",
    "licence / IP (licensor, licensee)": r"\b(licensor|licensee|intellectual property|patents?|trademarks?|royalt(y|ies))\b",
    "credit / loan (borrower, lender)": r"\b(borrower|lenders?|loans?|administrative agent|indebtedness|credit agreement)\b",
    "employment (employee, executive)": r"\b(employee|executive|employment|severance|base salary)\b",
    "M&A (buyer, seller, merger)": r"\b(purchaser|buyer|seller|merger|acquisition)\b",
    "securities (shares, underwriter)": r"\b(securities act|underwriters?|warrants?|common stock|shareholders?|stockholders?)\b",
}


def is_all_caps(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6


def scores(y, pred):
    present = np.unique(y)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds, names = load_ledgar()
    texts = list(ds["test"]["text"])
    y = np.array(ds["test"]["label"])

    # ---- predictions already saved in the repo
    preds, conf = {}, None
    preds["TF-IDF + LogReg"] = np.load("results/baseline/test_pred.npy")
    bert_probs = {}
    for s, T in SEEDS.items():
        path = f"results/legalbert_e10_s{s}/test_logits.npy"
        if os.path.exists(path):
            bert_probs[s] = softmax(np.load(path) / (T or 1.44), axis=1)
            preds[f"Legal-BERT s{s}"] = bert_probs[s].argmax(1)
    conf = np.mean([p.max(1) for p in bert_probs.values()], axis=0)
    if os.path.exists("models/baseline.joblib"):
        import joblib
        vec, clf = joblib.load("models/baseline.joblib")
        base_prob = clf.predict_proba(vec.transform(texts))
        for s in [1, 2]:
            if s in bert_probs:
                preds[f"Ensemble s{s}"] = ((bert_probs[s] + base_prob) / 2).argmax(1)
    else:
        print("note: models/baseline.joblib not found, ensemble rows skipped (run python -m src.baseline)")

    # ---- slices
    slices = {"all test clauses": np.ones(len(y), bool)}
    for fam, pattern in FAMILIES.items():
        rx = re.compile(pattern, re.IGNORECASE)
        slices[fam] = np.array([bool(rx.search(t)) for t in texts])
    slices["none of these families"] = ~np.any([slices[f] for f in FAMILIES], axis=0)
    slices["ALL-CAPS boilerplate"] = np.array([is_all_caps(t) for t in texts])

    rows = []
    for sl, m in slices.items():
        if not m.any():
            print(f"note: no test clause in slice '{sl}', skipped")
            continue
        row = {"slice": sl, "n": int(m.sum()), "n_types": int(len(np.unique(y[m]))),
               "legalbert_mean_conf": float(conf[m].mean()) if m.any() else None}
        for model, p in preds.items():
            if m.any():
                sc = scores(y[m], p[m])
                row[f"{model} | acc"] = sc["accuracy"]
                row[f"{model} | macro_f1"] = sc["macro_f1"]
        bert_cols = [c for c in row if c.startswith("Legal-BERT") and c.endswith("acc")]
        row["Legal-BERT mean | acc"] = float(np.mean([row[c] for c in bert_cols]))
        row["Legal-BERT mean | macro_f1"] = float(np.mean([row[c.replace("acc", "macro_f1")] for c in bert_cols]))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("slice")
    df.to_csv(f"{OUT_DIR}/slices.csv")
    json.dump({"families_regex": FAMILIES, "temperatures": {str(k): v for k, v in SEEDS.items()},
               "rows": df.reset_index().to_dict(orient="records")}, open(f"{OUT_DIR}/slices.json", "w"), indent=2)

    view_cols = ["n", "n_types", "TF-IDF + LogReg | acc", "Legal-BERT mean | acc", "Legal-BERT mean | macro_f1",
                 "legalbert_mean_conf"] + [c for c in df.columns if c.startswith("Ensemble") and c.endswith("acc")]
    pd.set_option("display.width", 220)
    print(df[view_cols].round(3).to_string())
    print(f"\nSaved {OUT_DIR}/slices.csv and slices.json")


if __name__ == "__main__":
    main()
