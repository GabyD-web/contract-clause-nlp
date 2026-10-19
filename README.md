# Contract Clause NLP: classification, retrieval and audit

NLP core of a contract-review assistant for a European legal-tech start-up. It:

1. **classifies** each contract clause into one of 100 clause types;
2. **retrieves** similar clauses from a clause bank, given a pasted clause or a lawyer's question;
3. **documents** where it can and cannot be trusted (evaluation, bias audit, model card).

## Data

[LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue) from the LexGLUE benchmark: clauses from US SEC (EDGAR) contracts, English, 100 clause types. Official splits: 60,000 train / 10,000 validation / 10,000 test.

Key findings from the EDA (`notebooks/01_eda.ipynb`):
- **Strong imbalance:** Governing Laws has 3,167 training examples, Books has 23 (ratio 138). Macro-F1 is therefore the headline metric.
- **Rare classes are barely measurable:** Books has 0 validation and 2 test examples.
- **Clause length:** median 101 tokens; 12.6% exceed 256 tokens, concentrated in a few types (68% of Capitalization clauses).
- **Leakage:** 282 test clauses (2.8%) also appear in train. Results are reported with and without them.
- **Label noise:** identical texts carry different labels (e.g. Waivers / No Waivers).

## Results

| Model | Test accuracy | Test macro-F1 (95% CI) | Macro-F1, dedup test | ECE |
|---|---|---|---|---|
| TF-IDF + logistic regression (C=30, tuned on validation) | 0.878 | 0.827 | 0.823 | *to compute* |
| BERT-base | *in progress* | | | |
| Legal-BERT (10 epochs, seed 42) | 0.884 | 0.830 [0.814 – 0.840] | 0.826 | 0.084 |
| Legal-BERT (3 seeds, mean ± std) | *in progress* | | | |

On LEDGAR, fine-tuned Legal-BERT improves accuracy by 0.7 points over a well-tuned TF-IDF model, but the macro-F1 gain (+0.3) lies within the confidence interval. Full history: [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md).

## Repository structure

```
configs/     training configurations (one YAML per model)
notebooks/   EDA, Colab runner, audit analysis
src/         data loading, training, evaluation, retrieval, audit
scripts/     run_all.sh reproduces every result
results/     metrics (JSON/CSV) and figures
docs/        model card, retrieval queries, out-of-scope samples
tests/       sanity tests
```

Inputs for the retrieval evaluation and the audit live in `docs/`:
- `docs/queries.csv` — 30 lawyer queries (10 frequent, 10 mid-frequency, 10 rare clause types) with the expected label and, for most, a string the top result must contain.
- `docs/out_of_scope.txt` — 50 out-of-scope samples (news, email chatter, legal-but-not-clause, French) for the abstention analysis.

The 50 most confident errors are catalogued in `results/errors_top50.csv`, with a manual category (`label_noise`, `overlapping_labels`, `multi_topic`, `model_error`) and a comment.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.baseline            # CPU, ~30 min
```

Transformer training runs on Google Colab (T4 GPU): see `notebooks/00_colab_runner.ipynb`.

## Reproducibility

Fixed seeds, official splits, configs in `configs/`, hyperparameters chosen on validation only.

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL.
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.