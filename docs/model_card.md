# Model card: contract clause classifier and clause-bank retrieval

## Model details
- **Classifier:** Legal-BERT (`nlpaueb/legal-bert-base-uncased`) fine-tuned on LEDGAR, 100 clause types,
  10 epochs, max 256 tokens, trained **six times**: seeds 42 / 1 / 2 on Colab and Kaggle T4 GPUs
  (`results/legalbert_e10_s*/`) and the same three seeds on a local RTX 4070 Ti SUPER
  (`results/local_gpu/legalbert_e10_s*/`).
- **Retrieval:** BM25 over the clause bank, compared against sentence-transformer embeddings
  (all-MiniLM-L6-v2) and a hybrid of the two.
- **Baseline:** TF-IDF + logistic regression (C tuned on validation), `results/baseline/`.
- **Calibration and ensemble:** temperature scaling fitted on validation, and a 50/50 probability
  ensemble of calibrated Legal-BERT with TF-IDF. See *Calibration, abstention and ensemble*.
- **Version:** v1.0, October 2026.
- **Authors:** Gaby, Theodore, Leopold.

### Run sets: which run lives where
These directories hold different runs and must not be mixed; each pair has the same seed but a
different machine.

| Directory | Runs | Contains | Used by |
|---|---|---|---|
| `results/legalbert_e10_s{42,1,2}/` | Colab / Kaggle T4 | metrics, `test_logits.npy`, `val_logits.npy` (s1, s2) | `notebooks/02_evaluation.ipynb`: significance tests, confusions, calibration, ensemble |
| `results/local_gpu/legalbert_e10_s{42,1,2}/` | local RTX 4070 Ti SUPER | metrics, per-class, confusions, train log | the bias audit (`results/audit/seed_*/`) |
| `results/baseline/` | reference baseline | metrics, `test_pred.npy` | McNemar baseline for the notebook |

Scores differ by up to 0.6 pt of macro-F1 between machines for the same seed (GPU kernels are
not bit-for-bit deterministic), so every headline number is a mean over six runs.

## Intended use
The model helps qualified lawyers during due diligence on English-language commercial contracts.
It suggests the type of each clause and shows similar clauses from the clause bank.

A lawyer reviews every output. Clauses below the confidence threshold are sent to a lawyer
(see *Abstention* below).

## Out-of-scope uses: the model must NOT be used for
- Legal advice, or deciding whether a clause is valid, enforceable or risky
- Contracts not written in English (French accuracy: not measured — see the audit;
  the model has no French training data and its French behaviour is untested)
- Consumer, employment or court documents without a new evaluation
- Fully automatic tagging without human review
- Any decision about a natural person

## Training data
LEDGAR (LexGLUE benchmark): 80,000 clauses from contracts filed with the US SEC
(60,000 train / 10,000 validation / 10,000 test).

- English only, US law, large public companies.
- Labels come from the clause headings. Because of this, some labels overlap
  (e.g. "Waivers" and "No Waivers") and some labels are wrong.
- The classes are very imbalanced. On the test set, Governing Laws has 582 clauses and Books has 2.

## Evaluation

### Classification (test set, 10,000 clauses, six runs on two machines)

| Model | Accuracy | Macro-F1 | Macro-F1 (de-duplicated) | ECE |
|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.8775 | 0.8267 | 0.8230 | 0.020 |
| Legal-BERT, Colab/Kaggle T4, 3 seeds | 0.8848 ± 0.0023 | 0.8286 ± 0.0040 | 0.8250 ± 0.0041 | 0.084 ± 0.001 |
| Legal-BERT, local GPU, 3 seeds | 0.8842 ± 0.0020 | 0.8254 ± 0.0020 | 0.8219 ± 0.0020 | 0.083 ± 0.002 |
| **Legal-BERT, all 6 runs** | **0.8845 ± 0.0019** | **0.8270 ± 0.0033** | 0.8235 ± 0.0033 | 0.083 ± 0.001 |

Mean ± sample standard deviation across runs. Baseline per-class results: `results/baseline/per_class.csv`.

- **Legal-BERT is worse calibrated than the baseline.** ECE 0.083 against 0.020 for TF-IDF:
  the transformer is roughly four times as overconfident. This matters more than the 0.7-point
  accuracy gain, because it is what the abstention threshold relies on. Temperature scaling on
  validation corrects it (ECE 0.083 → 0.024–0.028); see *Calibration, abstention and ensemble*.
- **Macro-F1: no difference between the two models.** The six-run mean (0.8270) equals the
  baseline (0.8267); individual runs fall on both sides of it (0.8233 – 0.8318). Paired bootstrap
  on seed 42: +0.3 pt, 95% CI [−0.7, +1.3]. Training on validation only ever showed a small edge
  (0.8325 vs 0.8185) which did not survive on test.
- **Accuracy: Legal-BERT is better, in every run.** All six runs score between 0.8823 and 0.8873,
  against 0.8775 for the baseline (+0.7 pt). On seed 42 the difference is significant
  (McNemar exact test, p = 0.012).
- **The macro-F1 gain, where it exists, is on frequent clause types.** Over TF-IDF (seed 42):
  +1.3 pt on the 20 most frequent types, +0.5 on the next 40, −0.4 on the 40 rarest
  (`results/f1_by_bucket.csv`). Fine-tuning helps where data is plentiful; on rare types the
  keyword model is as good.
- It is weaker on rare clause types. Most errors are between near-synonym labels.
  - The baseline's lowest F1 scores are on near-synonym labels: Applicable Laws (0.30, confused with Governing Laws), Jurisdictions (0.38), Venues (0.49), Miscellaneous (0.59), Agreements (0.59) and Modifications (0.60, confused with Amendments).
  - Legal-BERT does not fix these: the same labels dominate its confusion matrix.
    The top confusions are Applicable Laws → Governing Laws (33), Definitions → Defined Terms (24),
    Tax Withholdings → Withholdings (24), Authorizations → Authority (16), Successors → Binding Effects (16).
- No extra memorisation: removing the 282 duplicated test clauses costs both models the same 0.3 – 0.4 pt.
- Confidence threshold: see *Abstention*.

Per-class results: `results/baseline/per_class.csv` (baseline, reference run) and
`results/local_gpu/legalbert_e10_s{42,1,2}/per_class.csv` (local runs). Colab/Kaggle per-class
files sit in `results/legalbert_e10_s{42,1,2}/per_class.csv`.
Fifty most confident errors, hand-categorised: `results/errors_top50.csv` (full copy in
`results/error_taxonomy.csv`).
Of those 50: 25 are label noise, 12 overlapping labels, 11 multi-topic clauses and only 2 are
genuine model errors. **Half of the model's most confident mistakes are wrong labels, not wrong predictions** —
this is the single largest correction to the measured error.

### Retrieval (30 lawyer queries, `docs/queries.csv`)
The 30 queries cover 10 frequent, 10 mid-frequency and 10 rare clause types. 4 of them contain numbers.
The clause bank is the 60,000-clause training split; the test split is never indexed.

| Method | Hit@5 (correct label) | MRR | must_contain found in top 5 |
|---|---|---|---|
| BM25 only | **0.967** | **0.856** | 1.000 |
| Embeddings + FAISS only | 0.900 | 0.786 | 0.963 |
| Hybrid (BM25 + embeddings) | 0.933 | 0.832 | 1.000 |

- **BM25 alone is the best method.** The hybrid was expected to win; it does not. Fusing dense
  and lexical scores was tested with reciprocal rank fusion and with weighted score fusion at
  α ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}; none beat pure BM25 (best hybrid: MRR 0.8556 vs BM25 0.8556
  at α = 0.0, i.e. BM25 with extra machinery). Dense retrieval pulls in clauses that share a
  *topic* but not a *clause type* — the same near-synonym problem that dominates classifier errors.
- All the difficulty is in rare clause types: hit@5 is 1.000 for every method on frequent and
  mid-frequency queries, but 0.917 (BM25), 0.750 (embeddings) and 0.833 (hybrid) on rare ones.
- must_contain does not discriminate between methods (0.96 – 1.00). It is reported because it
  checks that numbers and key terms survive retrieval, which rank-based metrics cannot see.

30 queries is a small test set. One query is worth 3.3 points of Hit@5.

### Audit findings

Source: `results/audit/seed_{42,1,2}/summary.csv`, computed on the **three local-GPU models**
(`results/local_gpu/legalbert_e10_s*/`). The audit was run on Legal-BERT alone, not on the
calibrated model or the ensemble.

| Slice | n | Accuracy | Macro-F1 | Gap vs overall | Mean confidence |
|---|---|---|---|---|---|
| All test clauses | 10,000 | 0.884 | 0.825 | 0.000 | 0.967 |
| Frequent clause types (top 20) | 4,895 | 0.933 | 0.912 | +0.048 | 0.977 |
| Mid-frequency clause types (next 40) | 3,480 | 0.865 | 0.865 | −0.019 | 0.962 |
| Rare clause types (bottom 40) | 1,625 | 0.780 | 0.742 | −0.104 | **0.947** |
| Clauses longer than 256 tokens | 1,083 | **0.903** | 0.834 | **+0.019** | 0.972 |
| Clauses of 256 tokens or fewer | 8,917 | 0.882 | 0.826 | −0.002 | 0.966 |
| Non-clause English text (`docs/out_of_scope.txt`) | 41 | n/a | n/a | n/a | **0.47** (median 0.38) |
| French text (same file) | 9 | n/a | n/a | n/a | **0.48** (median 0.58) |

- **Truncation is not a problem, contrary to the EDA's concern.** Clauses longer than 256 tokens
  score slightly *better* than shorter ones (0.903 vs 0.882, consistent across all three seeds),
  so the type is decided early in the clause. The EDA noted 68% of Capitalization clauses are cut;
  in practice cutting them costs nothing measurable. The `max_length = 256` choice is validated.
- **The "fewer than 50 training examples" slice is too small to report.** It contains 11 test
  clauses spanning 3 classes and swings between 0.36 and 0.55 across seeds. It is replaced above
  by the bottom-40-class slice (n = 1,625), which is large enough to be meaningful.
- **Out-of-scope and French text are mostly, not fully, rejected.** Mean confidence 0.47 and 0.48
  against 0.97 on real clauses, so most are caught by a confidence threshold — but about 11% of both
  groups still exceed 0.9 confidence. A language and document-type check is needed in front of the
  model; it cannot be relied on to reject these by confidence alone.
- **Rare classes are overconfident.** The bottom-40 slice has 0.947 mean confidence at 0.780 accuracy —
  a 17-point confidence/accuracy gap, the worst calibration in any slice. This is the risk that
  matters in production: rare clauses are exactly where a lawyer needs the tool to say "I don't know".
- **Counterfactual audit (21–22 Oct, `src/counterfactual.py`).** Gender 0.2%, currency 1.1%, party names
  3.2–3.8% with no difference between names (chi-square p = 0.995, same 400 clauses), Governing Laws
  clauses 0–1.5% under US → EU swaps. A first run that sampled different clauses per name (3.2–5.8%, p = 0.47)
  was superseded.
- **Contract families (22 Oct, `src/slices.py`).** Lease / real estate is the weak spot (Legal-BERT 0.746,
  TF-IDF 0.749, n = 227); credit, M&A and licences around 0.90.
- **Reproduction run (21 Oct, RTX 4070 Ti SUPER).** Within 0.05–0.12 pt of the reference; baseline within
  0.1 pt. `results/rerun_2026-10-21/`.


### Calibration, abstention and ensemble

Computed in `notebooks/02_evaluation.ipynb` on the Kaggle seeds 1 and 2, the two runs whose
validation logits are saved (`results/legalbert_e10_s{1,2}/val_logits.npy`). Temperatures and
thresholds are fitted on **validation** only.

| Model (test set) | Accuracy | Macro-F1 | ECE | Auto-tagged at ~97% accuracy* |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.878 | 0.827 | 0.020 | 72.6% (acc. 96.5%) |
| Legal-BERT, calibrated (s1 / s2) | 0.887 / 0.883 | 0.832 / 0.824 | 0.024 / 0.028 | 58.0% / 65.3% |
| **Ensemble: calibrated Legal-BERT + TF-IDF, 50/50 (s1 / s2)** | **0.895 / 0.891** | **0.844 / 0.836** | **0.017 / 0.013** | **76.6% / 76.5%** (acc. 96.7%) |

\*Lowest confidence threshold whose **validation** accuracy reaches 97%, applied to the test set.

- **Calibration works.** Temperature scaling (T = 1.45 and 1.43) cuts test ECE from 0.083–0.085 to
  0.024–0.028 and test NLL from 0.62–0.64 to 0.52–0.53, close to logistic regression (0.020).
  After scaling, a displayed confidence of 0.90 really means about 90% correct.
- **Alone, Legal-BERT is no better at knowing when it is right.** At equal coverage, calibrated
  Legal-BERT and TF-IDF are as accurate on their most confident clauses (top 50%: 97.6% vs 98.0%;
  top 90%: 93.2% vs 92.4%).
- **The ensemble is the best configuration in the project.** Equal-weight probability averaging
  (nothing fitted) gives macro-F1 +1.2 pt over Legal-BERT alone, 95% CI [+0.6, +2.0] (s1) and
  [+0.7, +1.7] (s2), McNemar p < 10⁻⁷, with ECE 0.013–0.017. It tags about 77% of clauses
  automatically at about 97% accuracy; the other 23% go to a lawyer. The gain is confirmed on
  validation (macro-F1 0.8325 / 0.8365 vs 0.8248 / 0.8230), which is the check that would have
  selected it beforehand. TF-IDF adds almost no compute.
- **Raw-confidence abstention (superseded).** The original single-model sweep
  (`results/audit/seed_*/abstention.csv`, seed 42) reached only 82% coverage at 94.0% accuracy for
  a 0.95 threshold, and 93.3% at 91.7% for 0.85: because the model is overconfident, raw confidence
  alone does not reach 97% accuracy. It is kept for reference and replaced by the calibrated
  ensemble above, which is the configuration recommended for use.

Files: `results/abstention.json`, `results/abstention_table.csv`, `results/ensemble.json`,
`results/ensemble_summary.csv`, `results/figures/abstention_curve.png`.

## Limitations and risks
- The training contracts use US-centric drafting. The model has not been tested on EU contracts.
- Clauses longer than 256 tokens are cut. Measured effect: none on aggregate accuracy (+0.019 on the
  truncated slice); very long clauses (>512 tokens, 1.8% of the data) were not evaluated separately.
- Rare clause types (bottom 40 by frequency) are unreliable (0.780 accuracy) and overconfident (0.947).
- Labels come from clause headings, so they are noisy. **Of the 50 most confident errors, 48 are
  label noise, overlapping labels or multi-topic clauses, and only 2 are genuine model errors.**
  A large part of the measured error ceiling is the dataset, not the model.
- Raw Legal-BERT confidence cannot be used as a routing signal without calibration (ECE 0.083).
  Use the calibrated ensemble, or calibrate on your own validation data.
- Non-clause and French text is mostly rejected by confidence but not reliably: about 11% of both
  pass a 0.9 threshold. Put a language and document-type check before the model.
- Retrieval is BM25-based; it relies on lexical overlap and will miss paraphrases that share no words
  with the bank. Its weakness is concentrated in rare clause types (hit@5 = 0.917).
- Retrieval evaluation rests on 30 queries, which is too few for tight confidence intervals.
- The slice audit (rare types, length, out-of-scope text) was run on Legal-BERT alone; the counterfactual
  and contract-family audits cover TF-IDF, Legal-BERT and the ensemble.
- **Lease and real-estate clauses are a weak spot:** 74.6% accuracy at 86% confidence (n = 227), and
  TF-IDF is no better. Do not use on property or lease portfolios without adding lease contracts to training.
- Replacing a role (“the Executive”) with a person’s name changes the predicted type for 3–4% of clauses.
  Names of different origins give the same rate (chi-square p = 0.995), so no name-origin bias was found.
  Gender and currency swaps change ≤ 1.2%; US → EU governing-law swaps change ≤ 1.5% of Governing Laws
  clauses (ensemble ≤ 1.0%).

## Data protection (GDPR)
Client contracts can contain personal data. Run the model and the clause bank on the client's
own infrastructure or in an EU region. Do not use client data to retrain without written agreement.

## Reproducibility
- Code tag: `v1.0`
- Entry point: `scripts/run_all.sh`
- Dependencies: `requirements.lock.txt`, `requirements.txt`
- Seeds: 42 / 1 / 2, on two machines (Colab/Kaggle T4 and a local RTX 4070 Ti SUPER), six runs total.
  Export `CLAUSE_NLP_ROOT=results/local_gpu` to write local runs into
  `results/local_gpu/legalbert_e10_s*/` instead of the Colab Drive paths kept in `configs/`.
- The same seed gives up to 0.6 pt different macro-F1 across machines (seed 1: 0.8318 on Kaggle vs
  0.8256 locally; seed 42: 0.8298 vs 0.8273; seed 2: 0.8240 vs 0.8233), because GPU kernels are not
  bit-for-bit deterministic. That spread is as large as any difference between the models, which
  is why headline numbers are means over six runs. The baseline reproduces within 0.1 pt across machines.
- Timing: ~13 min per epoch on a Colab/Kaggle T4, ~4 min per epoch on a local RTX 4070 Ti SUPER
  (about 40 min for 10 epochs).
