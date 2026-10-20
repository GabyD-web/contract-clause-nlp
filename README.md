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

| Model | Test accuracy | Test macro-F1 (95% CI) | Test micro-F1 | Macro-F1, dedup test | ECE |
|---|---|---|---|---|---|
| TF-IDF + logistic regression (C=30, tuned on validation) | 0.877 | 0.827 | 1.000 | 0.823 | 0.021 |
| Legal-BERT, seed 42 (10 epochs) | 0.886 | 0.827 [0.813 – 0.836] | 1.000 | 0.824 | 0.081 |
| Legal-BERT (3 seeds, mean ± std) | 0.884 ± 0.002 | 0.825 ± 0.002 | — | 0.822 ± 0.002 | 0.083 ± 0.002 |

On LEDGAR, fine-tuning Legal-BERT does **not** improve macro-F1 over a well-tuned TF-IDF model:
the three-seed mean (0.8254) is marginally below the baseline (0.8267), a difference of −0.001 that
sits inside both the seed spread (±0.002) and the bootstrap CI. Accuracy is consistently higher
(+0.7 points, McNemar p = 0.012), and the transformer is markedly worse calibrated
(ECE 0.083 vs 0.021). Full history: [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md).

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

## Retrieval and audit

Retrieval on 30 lawyer queries (`docs/queries.csv`) against the 60,000-clause training bank:

| Method | Hit@5 | MRR | must_contain in top 5 |
|---|---|---|---|
| BM25 | **0.967** | **0.856** | 1.000 |
| Embeddings (all-MiniLM-L6-v2) | 0.900 | 0.786 | 0.963 |
| Hybrid (BM25 + embeddings) | 0.933 | 0.832 | 1.000 |

BM25 alone wins; fusion never beat it at any weight tested. Dense retrieval returns clauses that
share a topic but not a clause type. All the difficulty is in rare clause types (hit@5 0.917 vs 1.000).

Audit highlights (`results/audit/`, mean over 3 seeds):

- Frequent clause types: 0.933 accuracy. Rare (bottom 40 classes): 0.780.
- Clauses truncated at 256 tokens score **+0.019 higher** than short ones — truncation costs nothing measurable.
- Non-clause and French text are rejected: mean confidence 0.38 and 0.34 against 0.97 on real clauses.
- Rare classes are overconfident (0.948 confidence at 0.780 accuracy) — the worst calibration in any slice.
- At a 0.85 threshold, 82% of clauses are tagged automatically at 94.0% accuracy.

## Reproducibility

Fixed seeds, official splits, configs in `configs/`, hyperparameters chosen on validation only.

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL.
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.