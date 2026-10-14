from datasets import load_dataset

def load_ledgar():
    """LexGLUE LEDGAR: official train / validation / test splits. Never re-split."""
    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    return ds, names