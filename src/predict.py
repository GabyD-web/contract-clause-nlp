import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(**batch).logits
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)

class ClauseClassifier:
    """Raw text in, probabilities out. Used by the audit and the demo."""
    def __init__(self, model_dir, max_length=256, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_length = max_length
        self.names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)