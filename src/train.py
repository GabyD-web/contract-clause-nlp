import argparse, os, time
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score
from src.data import load_ledgar
from src.predict import predict_loader
from src.utils import set_seed, load_config, save_json

def main(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, names = load_ledgar()
    if cfg.get("max_train"):   # smoke test: train on a small subset
        ds["train"] = ds["train"].select(range(cfg["max_train"]))

    # --- tokenisation ---
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=cfg["max_length"])
    ds = ds.map(encode, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    ds.set_format("torch")
    collate = DataCollatorWithPadding(tok)   # dynamic padding: faster than padding to max_length
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(ds["train"], batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_dl = DataLoader(ds["validation"], batch_size=cfg["batch_size"] * 2, collate_fn=collate)

    # --- model, optimiser, schedule ---
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(names),
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_dl) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time(); running = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {running / step:.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step")

        logits, labels = predict_loader(model, val_dl, device)
        preds = logits.argmax(-1)
        row = {"epoch": epoch, "train_loss": running / len(train_dl),
               "val_micro_f1": f1_score(labels, preds, average="micro"),
               "val_macro_f1": f1_score(labels, preds, average="macro"),
               "epoch_minutes": (time.time() - t0) / 60}
        history.append(row); print(row)

        if row["val_macro_f1"] > best:
            best, bad_epochs = row["val_macro_f1"], 0
            model.save_pretrained(cfg["output_dir"]); tok.save_pretrained(cfg["output_dir"])
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print("early stopping"); break

    save_json({"config": cfg, "history": history, "best_val_macro_f1": best},
              os.path.join(cfg["results_dir"], "train_log.json"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:   # run several seeds from one config
        cfg["seed"] = args.seed
        cfg["output_dir"] = cfg["output_dir"].rsplit("_s", 1)[0] + f"_s{args.seed}"
        cfg["results_dir"] = cfg["results_dir"].rsplit("_s", 1)[0] + f"_s{args.seed}"
    main(cfg)