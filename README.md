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

**Cost of each model.**

| Model | Trainable parameters | Training time (one run) | Hardware | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Always “Governing Laws” (floor) | 0 | none | – | 0.058 | 0.001 |
| TF-IDF + logistic regression | 17.3 M (173,477 n-grams × 100 types) | 12 min (C=30; 29 min for the 3-value grid) | laptop CPU | 0.878 | 0.827 |
| Legal-BERT, 10 epochs | ≈110 M | 2 h 00 – 2 h 08 (T4) / 39 – 42 min (RTX 4070 Ti SUPER) | GPU | 0.885 | 0.827 |
| Ensemble (calibrated Legal-BERT + TF-IDF) | 1 extra (the temperature) | none beyond the two models | – | 0.891 – 0.895 | 0.836 – 0.844 |

Training times come from `results/*/train_log.json` (validation included) and
`results/baseline/test_metrics.json`. Legal-BERT has about 6× more parameters than the baseline and
needs about 10× its training time on a GPU, for +0.7 pt accuracy and no macro-F1 gain on its own;
combined with TF-IDF, which adds almost nothing at inference, it gives the only significant macro-F1 gain.

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
  The slice audit below (rare types, length, out-of-scope text) was run on Legal-BERT alone; the counterfactual and contract-family audits cover TF-IDF, Legal-BERT and the ensemble.

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

### Parties and kinds of contract

**Counterfactual swaps** (`src/counterfactual.py`, Legal-BERT seed 42 on the local GPU, temperature 1.44;
`results/audit/counterfactual.csv`). One detail of each clause is changed; a fair model keeps the clause
type. Every name and every jurisdiction is tested on the same clauses.

| Change | n | Type changes: Legal-BERT | TF-IDF | Ensemble |
|---|---|---|---|---|
| Gender: he → she | 400 | 0.2% | 1.2% | 1.2% |
| Party: “the Executive” / “the Employee” → 5 names | 5 × 400 | 3.2–3.8% | 1.8–2.0% | 2.2–3.8% |
| Governing law: US → Germany / France / England | 400 | 0.8–3.8% | 2.2% | 1.0–1.2% |
| … Governing Laws clauses only | 203 | 0.0–1.5% | 1.5–2.0% | 0.0–1.0% |
| Currency: $ → € | 376 | 1.1% | 0.0% | 0.8% |

- **No effect of the name’s origin.** On the same 400 clauses, John Smith, Marie Dubois, Fatou Diallo,
  Mehmet Yilmaz and Wei Zhang give the same flip rate (chi-square p = 0.995 for Legal-BERT, 0.999 for
  TF-IDF, 0.718 for the ensemble). The 3–4% that change come from replacing a role word with any
  person’s name: wording sensitivity, not origin. A first run sampled different clauses per name and
  showed 3.2–5.8% (p = 0.47); that spread was sampling noise.
- **Gender and currency are neutral** (≤ 1.2%). **Governing Laws clauses keep their type** in at least
  98.5% of US → EU swaps; other clauses that name a US state (e.g. “a Delaware corporation”) change up
  to 3.8%.

**Kinds of contract** (`src/slices.py`; families inferred from keywords, a clause can be in several;
Legal-BERT = mean of the Colab/Kaggle seeds; `results/audit/slices.csv`).

| Kind of contract | n | Types | TF-IDF acc. | Legal-BERT acc. | Legal-BERT macro-F1 | Legal-BERT confidence | Ensemble acc. (s1 / s2) |
|---|---|---|---|---|---|---|---|
| All test clauses | 10,000 | 100 | 0.878 | 0.885 | 0.829 | 0.911 | 0.895 / 0.891 |
| Credit / loan | 2,132 | 89 | 0.906 | 0.909 | 0.848 | 0.926 | 0.921 / 0.917 |
| M&A | 763 | 80 | 0.907 | 0.918 | 0.833 | 0.926 | 0.925 / 0.931 |
| Licence / IP | 186 | 42 | 0.903 | 0.900 | 0.771 | 0.918 | 0.903 / 0.909 |
| Securities | 774 | 81 | 0.880 | 0.882 | 0.797 | 0.914 | 0.891 / 0.891 |
| Employment | 1,749 | 86 | 0.854 | 0.863 | 0.716 | 0.904 | 0.873 / 0.874 |
| **Lease / real estate** | 227 | 64 | **0.749** | **0.746** | **0.654** | 0.861 | 0.762 / 0.767 |
| ALL-CAPS boilerplate | 236 | 20 | 0.894 | 0.897 | 0.541 | 0.884 | 0.907 / 0.898 |

- **Leases and real estate are the weak spot:** 74.6% accuracy at 86% confidence. TF-IDF is equally weak,
  so the training data lacks lease contracts; it is not a model problem.
- **ALL-CAPS boilerplate** keeps its accuracy but not its macro-F1 (0.541 over 20 types): frequent
  uppercase types are right, rare ones wrong.
- The ensemble is better than Legal-BERT alone in every family. Families come from keywords, so they
  overlap and differ in their mix of clause types; macro-F1 is computed over the types present.

The model card ([`docs/model_card.md`](docs/model_card.md)) lists intended uses, uses the model
must not be put to, and data-protection requirements.

## Repository structure

```
configs/     training configurations (one YAML per model)
notebooks/   EDA, Colab runner, evaluation analysis
src/         data loading, training, evaluation, retrieval, audit
scripts/     quickstart.sh (one-command setup); run_all.sh reproduces every result
results/     metrics (JSON/CSV), figures, experiment log
docs/        model card, retrieval queries, out-of-scope samples
tests/       sanity tests
```

## Quickstart

One command, on Linux, macOS, WSL and Windows (Git Bash):

```bash
./scripts/quickstart.sh                   # full quickstart, ~30 min (baseline on CPU)
./scripts/quickstart.sh --skip-baseline   # environment and tests only, ~2 min
```

It creates `.venv`, installs `requirements.txt`, runs the tests and the TF-IDF baseline, then
prints the test accuracy and macro-F1. It detects the platform and uses the right interpreter path
for the OS. If you run it inside a cloned repo it warns before overwriting `results/baseline/`,
which holds the reference run used by `notebooks/02_evaluation.ipynb`; set
`BASELINE_OUT=results/repro` to write somewhere else and keep the reference intact.

Or step by step:

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.baseline            # CPU, ~30 min
```

Expected baseline result: accuracy 0.8775, macro-F1 0.8267 (C=30). A rerun on a different
machine typically lands within 0.1 pt.

Transformer training needs a GPU:
- **Google Colab / Kaggle (T4):** `notebooks/00_colab_runner.ipynb`, about 13 min per epoch.
- **Local GPU:** `scripts/run_all.sh`, with `CLAUSE_NLP_ROOT` set to redirect the Colab Drive
  paths in the configs to a local folder.

## Reproducibility

Official splits, fixed seeds (42 / 1 / 2), configs in `configs/`, hyperparameters chosen on
validation only, pinned versions in `requirements.lock.txt`. Results are stable to within about
±0.5 pt across seeds and hardware; exact bit-for-bit reproduction across different GPUs is not
possible, so all headline numbers are means over six runs.

**Full reproduction run (21 Oct).** The whole pipeline (`scripts/run_all.sh`) was rerun on an RTX 4070 Ti
SUPER: the three Legal-BERT seeds land within 0.05–0.12 pt of the committed reference (accuracy
0.8854 ± 0.0005, macro-F1 0.8258 ± 0.0015 vs 0.8842 / 0.8254) and the baseline within 0.1 pt
(0.8767 / 0.8266 vs 0.8775 / 0.8267). Outputs are in `results/rerun_2026-10-21/`; the committed
reference files were left unchanged.

## References

- Chalkidis et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL.
- Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
