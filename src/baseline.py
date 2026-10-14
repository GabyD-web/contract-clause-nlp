import json, time, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.data import load_ledgar
from src.utils import set_seed, save_json

C_GRID = [1, 10, 30]   

def scores(y, p):
    return {"accuracy": accuracy_score(y, p),
            "micro_f1": f1_score(y, p, average="micro"),
            "macro_f1": f1_score(y, p, average="macro")}

def main():
    set_seed(42)
    ds, names = load_ledgar()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(ds["train"]["text"])
    Xva = vec.transform(ds["validation"]["text"])
    Xte = vec.transform(ds["test"]["text"])
    ytr, yva, yte = (np.array(ds[s]["label"]) for s in ["train", "validation", "test"])

    # 1. choose C on validation
    search = {}
    best_C, best_f1, best_clf = None, -1, None
    for C in C_GRID:
        t0 = time.time()
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        search[C] = {"val_macro_f1": f1, "train_seconds": time.time() - t0}
        print(f"C={C}: val macro-F1 {f1:.4f} ({search[C]['train_seconds']:.0f}s)")
        if f1 > best_f1:
            best_C, best_f1, best_clf = C, f1, clf

    # 2. evaluate the chosen model once on test (full and de-duplicated)
    pred = best_clf.predict(Xte)
    dup = set(json.load(open("results/test_duplicate_idx.json")))
    keep = np.array([i not in dup for i in range(len(yte))])
    res = {"model": "tfidf+logreg", "best_C": best_C, "C_search": search,
           "test": scores(yte, pred),
           "test_dedup": scores(yte[keep], pred[keep]),
           "n_test": int(len(yte)), "n_test_dedup": int(keep.sum())}
    print(json.dumps(res, indent=2))

    save_json(res, "results/baseline/test_metrics.json")
    np.save("results/baseline/test_pred.npy", pred)
    rep = classification_report(yte, pred, labels=list(range(len(names))), target_names=names,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv("results/baseline/per_class.csv")
    joblib.dump((vec, best_clf), "models/baseline.joblib")

if __name__ == "__main__":
    main()