import os, random, json
import numpy as np
import torch
import yaml

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def resolve_paths(cfg: dict) -> dict:
    """Rewrite Colab Drive paths for local runs.

    Configs keep the Colab path so the notebook runner works unchanged. Set
    CLAUSE_NLP_ROOT to train locally: the '/content/drive/MyDrive/clause-nlp/'
    prefix is replaced by that directory (relative to the CWD if not absolute).
    """
    root = os.environ.get("CLAUSE_NLP_ROOT")
    if not root:
        return cfg
    for key in ("output_dir", "results_dir"):
        p = cfg[key]
        if p.startswith("/content"):
            p = p.split("clause-nlp/", 1)[-1]
        cfg[key] = os.path.join(root, p)
    return cfg