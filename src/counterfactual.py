"""Counterfactual bias audit.

Question: does the predicted clause type change when we change ONLY who the parties are
(name, gender) or which country the contract belongs to (governing law, currency)?
A fair model keeps the same clause type: the type of a clause does not depend on these details.

For every test, we take the test clauses that contain the pattern, apply the change, and compare
predictions before and after, for Legal-BERT, TF-IDF + logistic regression and their ensemble.

Usage (repo root):
    python -m src.counterfactual --model_dir models/legalbert_e10_s1 --temperature 1.45
Outputs:
    results/audit/counterfactual.csv          one row per (test, model)
    results/audit/counterfactual.json         same + run configuration
    results/audit/counterfactual_examples.csv a few clauses whose type flipped, per test
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data import load_ledgar

OUT_DIR = "results/audit"


# ---------------------------------------------------------------- text transformations
def swap_words(mapping):
    """Whole-word, case-preserving replacement (he -> she, He -> She, HE -> SHE)."""
    rx = re.compile(r"\b(" + "|".join(map(re.escape, mapping)) + r")\b", re.IGNORECASE)

    def repl(m):
        w = m.group(0)
        r = mapping[w.lower()]
        if w.isupper():
            return r.upper()
        if w[0].isupper():
            return r[0].upper() + r[1:]
        return r

    return lambda t: rx.sub(repl, t)


def jurisdiction(country, adjective):
    """'(the) State of New York / Delaware' -> country (with its article); other mentions -> adjective."""
    state = re.compile(r"\b(?:([Tt])he )?[Ss]tate of (?:New York|Delaware)\b")
    bare = re.compile(r"\b(?:New York|Delaware)\b")

    def repl(m):
        out = country
        if m.group(1) == "T" and out.startswith("the "):     # keep a capital at sentence start
            out = "The " + out[4:]
        return out

    return lambda t: bare.sub(adjective, state.sub(repl, t))


def party(name):
    """'the Executive' / 'Executive' / 'the Employee' -> a named person.
    Capital E only, and not job titles such as 'Chief Executive Officer' or 'Executive Committee'."""
    rx = re.compile(r"\b(?:[Tt]he )?(?<!Chief )(?:Executive|Employee)\b"
                    r"(?!\s+(?:[Oo]fficer|[Cc]ommittee|[Vv]ice|[Dd]irector|[Cc]hairman|[Ss]ession|[Bb]enefit|[Cc]ompensation))")
    return lambda t: rx.sub(name, t)


def currency():
    sign = re.compile(r"\$\s?(?=\d)")
    words = re.compile(r"\b(?:U\.S\.|US|United States)\s+[Dd]ollars?\b|\b[Dd]ollars?\b")
    return lambda t: words.sub("euros", sign.sub("€", t))


NAMES = {"john_smith": "John Smith", "marie_dubois": "Marie Dubois", "fatou_diallo": "Fatou Diallo",
         "mehmet_yilmaz": "Mehmet Yilmaz", "wei_zhang": "Wei Zhang"}

TESTS = {
    "gender_he_to_she": ("gender", swap_words({"he": "she", "him": "her", "his": "her", "himself": "herself"})),
    **{f"party_{k}": ("party name", party(v)) for k, v in NAMES.items()},
    "jurisdiction_us_to_germany": ("jurisdiction", jurisdiction("the Federal Republic of Germany", "German")),
    "jurisdiction_us_to_france": ("jurisdiction", jurisdiction("the French Republic", "French")),
    "jurisdiction_us_to_england": ("jurisdiction", jurisdiction("England and Wales", "English")),
    "currency_usd_to_eur": ("currency", currency()),
}


# ---------------------------------------------------------------- models
class Predictor:
    """Caches probabilities per text so original clauses are only scored once."""

    def __init__(self, model_dir, temperature, max_length, baseline_path, batch_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.T, self.max_length, self.bs = temperature, max_length, batch_size
        self.tfidf = None
        if baseline_path and os.path.exists(baseline_path):
            import joblib
            self.tfidf = joblib.load(baseline_path)
        self.cache = {}

    @torch.no_grad()
    def _bert(self, texts):
        out = []
        for i in range(0, len(texts), self.bs):
            enc = self.tok(texts[i:i + self.bs], truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(self.model(**enc).logits.float().cpu().numpy())
        return softmax(np.concatenate(out) / self.T, axis=1)

    def probs(self, texts):
        """Returns {model_name: (n, 100) probabilities}."""
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            pb = self._bert(todo)
            pt = self.tfidf[1].predict_proba(self.tfidf[0].transform(todo)) if self.tfidf else None
            for j, t in enumerate(todo):
                self.cache[t] = (pb[j], None if pt is None else pt[j])
        bert = np.stack([self.cache[t][0] for t in texts])
        res = {"Legal-BERT (calibrated)": bert}
        if self.tfidf:
            tfidf = np.stack([self.cache[t][1] for t in texts])
            res["TF-IDF + LogReg"] = tfidf
            res["Ensemble"] = (bert + tfidf) / 2
        return res


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def compare(p0, p1, y):
    a0, a1 = p0.argmax(1), p1.argmax(1)
    flips = int((a0 != a1).sum())
    n = len(y)
    return {"n": n, "flips": flips, "flip_rate": flips / n if n else None, "flip_rate_ci95": wilson(flips, n),
            "acc_before": float((a0 == y).mean()), "acc_after": float((a1 == y).mean()),
            "correct_to_wrong": int(((a0 == y) & (a1 != y)).sum()),
            "conf_before": float(p0.max(1).mean()), "conf_after": float(p1.max(1).mean())}


# ---------------------------------------------------------------- main
def main(a):
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    ds, names = load_ledgar()
    texts = list(ds["test"]["text"])
    labels = np.array(ds["test"]["label"])
    gl = names.index("Governing Laws")
    rng = np.random.default_rng(a.seed)

    # which clauses each test applies to (only clauses the change actually modifies)
    plan = {}
    for test, (family, f) in TESTS.items():
        idx = [i for i, t in enumerate(texts) if f(t) != t]
        if len(idx) > a.max_per_test:
            idx = sorted(rng.choice(idx, a.max_per_test, replace=False).tolist())
        plan[test] = idx
        print(f"{test:32s} {len(idx):5d} clauses")
    n_unique = len({i for v in plan.values() for i in v})
    print(f"\n{n_unique} distinct original clauses, {sum(map(len, plan.values()))} modified clauses to score "
          f"on {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    pred = Predictor(a.model_dir, a.temperature, a.max_length, a.baseline, a.batch_size)
    if pred.tfidf is None:
        print(f"note: {a.baseline} not found, so only Legal-BERT is audited (run python -m src.baseline to add TF-IDF)")

    rows, examples = [], []
    for test, (family, f) in TESTS.items():
        idx = plan[test]
        if not idx:
            continue
        orig = [texts[i] for i in idx]
        new = [f(t) for t in orig]
        y = labels[idx]
        p0, p1 = pred.probs(orig), pred.probs(new)
        for model in p0:
            rows.append({"test": test, "family": family, "model": model, **compare(p0[model], p1[model], y)})
            if test.startswith("jurisdiction"):  # governing-law clauses must stay governing-law clauses
                m = y == gl
                if m.any():
                    rows.append({"test": test + " [Governing Laws only]", "family": family, "model": model,
                                 **compare(p0[model][m], p1[model][m], y[m])})
        b0, b1 = p0["Legal-BERT (calibrated)"].argmax(1), p1["Legal-BERT (calibrated)"].argmax(1)
        for j in np.where(b0 != b1)[0][:5]:
            examples.append({"test": test, "true": names[y[j]], "before": names[b0[j]], "after": names[b1[j]],
                             "original": orig[j][:400], "changed": new[j][:400]})
        print(f"done: {test} ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/counterfactual.csv", index=False)
    pd.DataFrame(examples).to_csv(f"{OUT_DIR}/counterfactual_examples.csv", index=False, encoding="utf-8-sig")

    names_rows = df[(df.family == "party name")]
    name_gap = (names_rows.groupby("model").flip_rate.agg(lambda s: float(s.max() - s.min()))).to_dict()
    with open(f"{OUT_DIR}/counterfactual.json", "w") as fh:
        json.dump({"config": {"model_dir": a.model_dir, "temperature": a.temperature, "max_length": a.max_length,
                              "max_per_test": a.max_per_test, "seed": a.seed,
                              "baseline": a.baseline if pred.tfidf else None},
                   "party_name_flip_rate_gap": name_gap,
                   "results": df.to_dict(orient="records")}, fh, indent=2)

    view = df.pivot_table(index="test", columns="model", values="flip_rate", sort=False)
    view.insert(0, "n", df.groupby("test", sort=False).n.first())
    pd.set_option("display.width", 200)
    print("\nFlip rate: share of clauses whose predicted type changes\n")
    print(view.round(3).to_string())
    print("\nLargest flip-rate gap between party names:", {k: round(v, 3) for k, v in name_gap.items()})
    print(f"\nSaved to {OUT_DIR}/counterfactual.csv, .json and counterfactual_examples.csv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="folder of a fine-tuned Legal-BERT run")
    ap.add_argument("--temperature", type=float, default=1.44, help="temperature fitted on validation")
    ap.add_argument("--baseline", default="models/baseline.joblib")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_per_test", type=int, default=400, help="cap per test, to keep CPU time short")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
