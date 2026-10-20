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

## Classification results (test set, 10,000 clauses)

Legal-BERT (`nlpaueb/legal-bert-base-uncased`, 10 epochs, lr 3e-5, max 256 tokens) was trained
**six times with the same configuration**: seeds 42 / 1 / 2 on Google Colab and Kaggle T4 GPUs
(`results/legalbert_e10_s*/`), and the same three seeds on a local RTX 4070 Ti SUPER
(`results/local_gpu/legalbert_e10_s*/`). All hyperparameters were chosen on validation; the test
set was used once per run.

| Model | Accuracy | Macro-F1 | Macro-F1, dedup test | ECE |
|---|---|---|---|---|
| TF-IDF + logistic regression (C=30, tuned on validation) | 0.8775 | 0.8267 | 0.8230 | 0.020 |
| Legal-BERT, Colab/Kaggle T4, 3 seeds | 0.8848 ± 0.0023 | 0.8286 ± 0.0040 | 0.8250 ± 0.0041 | 0.084 ± 0.001 |
| Legal-BERT, local GPU, 3 seeds | 0.8842 ± 0.0020 | 0.8254 ± 0.0020 | 0.8219 ± 0.0020 | 0.083 ± 0.002 |
| **Legal-BERT, all 6 runs** | **0.8845 ± 0.0019** | **0.8270 ± 0.0033** | 0.8235 ± 0.0033 | 0.083 ± 0.001 |

Mean ± sample standard deviation across runs. Single runs, the 95% bootstrap CI of each run and
the full training history are in [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md).

**What this shows**
- **Accuracy: Legal-BERT is better, in every run.** All six runs score between 0.8823 and 0.8873,
  against 0.8775 for the baseline (+0.7 pt). On seed 42 the difference is significant
  (McNemar exact test, p = 0.012).
- **Macro-F1: no difference.** The six-run mean (0.8270) equals the baseline (0.8267); individual
  runs fall on both sides of it (0.8233 – 0.8318). Paired bootstrap on seed 42: +0.3 pt,
  95% CI [−0.7, +1.3].
- **The gain is only on frequent clause types.** Macro-F1 gain over TF-IDF (seed 42): +1.3 pt on
  the 20 most frequent types, +0.5 on the next 40, −0.4 on the 40 rarest
  (`results/f1_by_bucket.csv`). Fine-tuning helps where data is plentiful; on rare types the
  keyword model is as good.
- **Most errors are between near-synonym labels**, often in both directions: Applicable Laws →
  Governing Laws, Definitions ↔ Defined Terms, Tax Withholdings ↔ Withholdings, Authority ↔
  Authorizations, Waivers ↔ No Waivers. Of the 50 most confident errors (hand-categorised in
  `results/errors_top50.csv`), **25 are label noise, 12 overlapping labels, 11 multi-topic clauses
  and only 2 genuine model errors.**
- **Legal-BERT is overconfident.** ECE 0.083 against 0.020 for logistic regression, so its raw
  confidence must be calibrated before it decides which clauses go to a lawyer
  (done below: ECE 0.083 → 0.024).
- **No extra memorisation.** Removing the 282 duplicated test clauses costs both models the same
  0.3 – 0.4 pt.

**Reproducibility across hardware.** The same seed and configuration gives up to 0.6 pt different
macro-F1 on different GPUs (seed 1: 0.8318 on Kaggle vs 0.8256 locally; seed 42: 0.8298 vs 0.8273;
seed 2: 0.8240 vs 0.8233), because GPU kernels are not bit-for-bit deterministic. That spread is
as large as any difference between the models, which is why we report the mean over six runs
rather than a single run. The baseline reproduces within 0.1 pt across machines.

Which runs each analysis uses:
- `notebooks/02_evaluation.ipynb`: significance tests, buckets and confusions use the Colab seed 42
  run (`results/legalbert_e10_s42/`); calibration, abstention and the ensemble use the Kaggle
  seeds 1 and 2, which also saved validation logits (`results/legalbert_e10_s{1,2}/`).
- The audit (`results/audit/seed_*/`) uses the three local-GPU models.

### Calibration, abstention and ensemble

Computed in `notebooks/02_evaluation.ipynb` on the Kaggle seeds 1 and 2, the two runs whose
validation logits are saved. Temperatures and thresholds are fitted on **validation** only.

| Model (test set) | Accuracy | Macro-F1 | ECE | Auto-tagged at ~97% accuracy* |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.878 | 0.827 | 0.020 | 72.6% (acc. 96.5%) |
| Legal-BERT, calibrated (s1 / s2) | 0.887 / 0.883 | 0.832 / 0.824 | 0.024 / 0.028 | 58.0% / 65.3% (acc. 97.2% / 96.8%) |
| **Ensemble: calibrated Legal-BERT + TF-IDF, 50/50 (s1 / s2)** | **0.895 / 0.891** | **0.844 / 0.836** | **0.017 / 0.013** | **76.6% / 76.5%** (acc. 96.7%) |

\*Lowest confidence threshold whose **validation** accuracy reaches 97%, applied to the test set.

- **Calibration works.** Temperature scaling (T = 1.45 and 1.43) cuts Legal-BERT's test ECE from
  0.083–0.085 to 0.024–0.028 and its test NLL from 0.62–0.64 to 0.52–0.53, close to logistic
  regression (0.020). After scaling, a displayed confidence of 0.90 really means about 90% correct.
- **Alone, Legal-BERT is no better at knowing when it is right.** At equal coverage, calibrated
  Legal-BERT and TF-IDF are as accurate on their most confident clauses (top 50%: 97.6% vs 98.0%;
  top 90%: 93.2% vs 92.4%). For deciding what goes to a lawyer, it brings no advantage on its own.
- **The ensemble beats both.** Averaging the two models' probabilities (equal weights, nothing
  fitted) gives the best accuracy, macro-F1 and calibration of the project. Against Legal-BERT
  alone: macro-F1 +1.2 pt, 95% CI [+0.6, +2.0] (seed 1) and [+0.7, +1.7] (seed 2); McNemar
  p < 10⁻⁷. It tags about 77% of test clauses automatically at about 97% accuracy; the other 23%
  go to a lawyer. The two models are complementary: Legal-BERT is stronger on frequent clause types,
  TF-IDF on rare keyword-defined ones. TF-IDF adds almost no compute.
- **How the ensemble was chosen.** The idea came after the single-model test results. It has no
  fitted parameters, and its gain is confirmed on validation (macro-F1 0.8325 / 0.8365 vs
  0.8248 / 0.8230 for Legal-BERT alone), which is the check that would have selected it beforehand.
  The bias audit below was run on Legal-BERT alone, not on the ensemble.

Files: `results/abstention.json`, `results/abstention_table.csv`, `results/ensemble.json`,
`results/ensemble_summary.csv`, `results/figures/abstention_curve.png`.

## Retrieval

Retrieval on 30 lawyer queries (`docs/queries.csv`: 10 frequent, 10 mid-frequency and 10 rare
clause types) against the 60,000-clause training bank. The test split is never indexed.

| Method | Hit@5 | MRR | must_contain in top 5 |
|---|---|---|---|
| BM25 | **0.967** | **0.856** | 1.000 |
| Embeddings (all-MiniLM-L6-v2 + FAISS) | 0.900 | 0.786 | 0.963 |
| Hybrid (BM25 + embeddings) | 0.933 | 0.832 | 1.000 |

- **BM25 alone is best.** Score fusion was tested at several weights and never beat it. Dense
  retrieval returns clauses that share a topic but not a clause type, the same near-synonym
  problem as in classification.
- **All the difficulty is in rare clause types:** hit@5 is 1.000 for every method on frequent and
  mid-frequency queries, but 0.917 (BM25), 0.750 (embeddings) and 0.833 (hybrid) on rare ones
  (`results/retrieval/by_bucket.csv`).
- 30 queries is a small test set: one query is worth 3.3 pt of hit@5.

## Audit

Mean over the three local-GPU models (`results/audit/seed_{42,1,2}/`).

| Slice | n | Accuracy | Macro-F1 | Mean confidence |
|---|---|---|---|---|
| All test clauses | 10,000 | 0.884 | 0.825 | 0.967 |
| Frequent clause types (top 20) | 4,895 | 0.933 | 0.912 | 0.977 |
| Mid-frequency types (next 40) | 3,480 | 0.865 | 0.865 | 0.962 |
| Rarest types (bottom 40) | 1,625 | 0.780 | 0.742 | **0.947** |
| Clauses longer than 256 tokens (truncated) | 1,083 | 0.903 | 0.834 | 0.972 |
| Clauses of 256 tokens or fewer | 8,917 | 0.882 | 0.826 | 0.966 |

- **Rare clause types are the main risk:** 0.780 accuracy at 0.947 mean confidence, the largest
  confidence/accuracy gap of any slice. These are the clauses where the tool should say "I don't know".
- **Truncation at 256 tokens costs nothing measurable:** truncated clauses score slightly higher
  (0.903 vs 0.882), so the type is decided early in the clause. The `max_length = 256` choice holds.
- **Out-of-scope text** (`docs/out_of_scope.txt`): 41 non-clause English sentences get a mean
  confidence of 0.47 (median 0.38) and 9 French sentences 0.48, against 0.97 on real clauses.
  Most are rejected by a confidence threshold, but about 11% of both groups still exceed 0.9
  confidence, so a language and document-type check is needed in front of the model.
- **Abstention (raw confidence, mean over 3 seeds, `results/audit/seed_*/abstention.csv`):** at a
  0.85 threshold the model tags 93.3% of clauses automatically at 91.7% accuracy; at 0.95, 88.5% at
  93.2%. Because the model is overconfident, even high raw thresholds do not reach 97% accuracy.
    After temperature scaling, and combined with TF-IDF, the system reaches about 97% accuracy on
  77% of clauses: see *Calibration, abstention and ensemble* above.

The model card ([`docs/model_card.md`](docs/model_card.md)) lists intended uses, uses the model
must not be put to, and data-protection requirements.

## Repository structure

```
configs/     training configurations (one YAML per model)
notebooks/   EDA, Colab runner, evaluation analysis
src/         data loading, training, evaluation, retrieval, audit
scripts/     run_all.sh reproduces every result
results/     metrics (JSON/CSV), figures, experiment log
docs/        model card, retrieval queries, out-of-scope samples
tests/       sanity tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.baseline            # CPU, ~30 min
```

Transformer training needs a GPU:
- **Google Colab / Kaggle (T4):** `notebooks/00_colab_runner.ipynb`, about 13 min per epoch.
- **Local GPU:** `scripts/run_all.sh`, with `CLAUSE_NLP_ROOT` set to redirect the Colab Drive
  paths in the configs to a local folder.

## Reproducibility

Official splits, fixed seeds (42 / 1 / 2), configs in `configs/`, hyperparameters chosen on
validation only, pinned versions in `requirements.lock.txt`. Results are stable to within about
±0.5 pt across seeds and hardware; exact bit-for-bit reproduction across different GPUs is not
possible, so all headline numbers are means over six runs.

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL.
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
