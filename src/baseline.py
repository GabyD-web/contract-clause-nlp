import time, joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.data import load_ledgar
from src.utils import set_seed, save_json

def main():
    set_seed(42)
    ds, names = load_ledgar()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(ds["train"]["text"])
    Xte = vec.transform(ds["test"]["text"])
    ytr, yte = np.array(ds["train"]["label"]), np.array(ds["test"]["label"])

    t0 = time.time()
    clf = LogisticRegression(C=10, max_iter=2000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    res = {
        "model": "tfidf+logreg",
        "accuracy": accuracy_score(yte, pred),
        "micro_f1": f1_score(yte, pred, average="micro"),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "train_seconds": time.time() - t0,
    }
    print(res)
    save_json(res, "results/baseline_metrics.json")
    np.save("results/baseline_test_pred.npy", pred)
    joblib.dump((vec, clf), "models/baseline.joblib")

if __name__ == "__main__":
    main()