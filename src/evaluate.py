import argparse, numpy as np, pandas as pd, torch, json, os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.data import load_ledgar
from src.predict import predict_loader
from src.utils import save_json

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    bins, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)

def bootstrap_ci(y, p, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed); scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        scores.append(metric(y[i], p[i]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]

def main(model_dir, out_dir, max_length):
    os.makedirs(out_dir, exist_ok=True)
    ds, names = load_ledgar()
    tok = AutoTokenizer.from_pretrained(model_dir)
    test = ds["test"].map(lambda b: tok(b["text"], truncation=True, max_length=max_length),
                          batched=True, remove_columns=["text"]).rename_column("label", "labels")
    test.set_format("torch")
    dl = DataLoader(test, batch_size=64, collate_fn=DataCollatorWithPadding(tok))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    logits, y = predict_loader(model, dl, device)
    np.save(f"{out_dir}/test_logits.npy", logits)

    probs = softmax(logits); p = probs.argmax(1)
    macro = lambda a, b: f1_score(a, b, average="macro")
    res = {"accuracy": accuracy_score(y, p),
           "micro_f1": f1_score(y, p, average="micro"),
           "macro_f1": macro(y, p),
           "macro_f1_95ci": bootstrap_ci(y, p, macro),
           "ece": ece(probs, y)}
    dup = set(json.load(open("results/test_duplicate_idx.json")))
    keep = np.array([i not in dup for i in range(len(y))])
    res["accuracy_dedup"] = accuracy_score(y[keep], p[keep])
    res["macro_f1_dedup"] = f1_score(y[keep], p[keep], average="macro")
    save_json(res, f"{out_dir}/test_metrics.json"); print(res)

    rep = classification_report(y, p, target_names=names, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"{out_dir}/per_class.csv")

    cm = confusion_matrix(y, p, labels=list(range(len(names))))
    np.fill_diagonal(cm, 0)
    pairs = [(names[i], names[j], int(cm[i, j])) for i, j in zip(*np.unravel_index(np.argsort(cm, None)[::-1][:20], cm.shape))]
    pd.DataFrame(pairs, columns=["true", "predicted", "count"]).to_csv(f"{out_dir}/top_confusions.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    a = ap.parse_args(); main(a.model_dir, a.out_dir, a.max_length)