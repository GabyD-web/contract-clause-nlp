# Model card: contract clause classifier and clause-bank retrieval

## Model details
- **Classifier:** Legal-BERT (`nlpaueb/legal-bert-base-uncased`) fine-tuned on LEDGAR, 100 clause types,
  10 epochs, max 256 tokens, 3 seeds (42 / 1 / 2).
- **Retrieval:** BM25 over the clause bank, compared against sentence-transformer embeddings
  (all-MiniLM-L6-v2) and a hybrid of the two.
- **Baseline:** TF-IDF + logistic regression (C tuned on validation).
- **Version:** v1.0, October 2026.
- **Authors:** Gaby, Theodore, Leopold.

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

### Classification (test set, 10,000 clauses, 3 seeds)

| Model | Accuracy | Macro-F1 | Macro-F1 (de-duplicated) | ECE |
|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.877 | 0.827 | 0.823 | 0.021 |
| Legal-BERT (mean ± std over 3 seeds) | 0.884 ± 0.002 | 0.825 ± 0.002 | 0.822 ± 0.002 | 0.083 ± 0.002 |
| Legal-BERT, seed 42 (single run) | 0.886 | 0.827 [0.813 – 0.836] | 0.824 | 0.081 |

- **Legal-BERT is worse calibrated than the baseline.** ECE 0.083 against 0.021 for TF-IDF:
  the transformer is roughly four times as overconfident. This matters more than the 0.7-point
  accuracy gain, because it is what the abstention threshold relies on.
- **Legal-BERT is not better than the keyword baseline on macro-F1.** The three-seed mean is
  0.8254 against 0.8267 for TF-IDF, a difference of −0.001 that is well inside the
  seed spread (±0.002) and inside the bootstrap CI. Training on validation only ever showed
  a small edge (0.8325 vs 0.8185), which did not survive on test.
- Accuracy is consistently higher for Legal-BERT (+0.7 points). A McNemar test on the
  seed-42 predictions is significant on accuracy (p = 0.012), so the accuracy gain is real
  even though the macro-F1 gain is not.
- It is weaker on rare clause types. Most errors are between near-synonym labels.
  - The baseline's lowest F1 scores are on near-synonym labels: Applicable Laws (0.30, confused with Governing Laws), Jurisdictions (0.38), Venues (0.49), Miscellaneous (0.59), Agreements (0.59) and Modifications (0.60, confused with Amendments).
  - Legal-BERT does not fix these: the same labels dominate its confusion matrix.
    The top confusions are Applicable Laws → Governing Laws (33), Definitions → Defined Terms (24),
    Tax Withholdings → Withholdings (24), Authorizations → Authority (16), Successors → Binding Effects (16).
- Confidence threshold: see *Abstention*.

Per-class results: `results/baseline/per_class.csv`, `results/legalbert_e10_s{42,1,2}/per_class.csv`.
Fifty most confident errors, hand-categorised: `results/errors_top50.csv`.
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

Two slices from the original plan are reported, and two are reported with caveats.
Source: `results/audit/seed_{42,1,2}/summary.csv` (mean over 3 seeds).

| Slice | n | Accuracy | Macro-F1 | Gap vs overall | Mean confidence |
|---|---|---|---|---|---|
| Frequent clause types (top 20) | 4,895 | 0.933 | 0.912 | +0.048 | 0.977 |
| Rare clause types, bottom 40 (<50–100 train ex.) | 1,625 | 0.780 | 0.742 | −0.104 | 0.948 |
| Clauses longer than 256 tokens | 1,083 | **0.903** | 0.834 | **+0.019** | 0.972 |
| Clauses of 256 tokens or fewer | 8,917 | 0.882 | 0.826 | −0.002 | 0.967 |
| Non-clause English text (`docs/out_of_scope.txt`) | 41 | n/a | n/a | n/a | **0.376** |
| French text (same file) | 9 | n/a | n/a | n/a | **0.342** |

- **Truncation is not a problem, contrary to the EDA's concern.** Clauses longer than 256 tokens
  score slightly *better* than shorter ones (0.903 vs 0.882, consistent across all three seeds).
  The EDA noted 68% of Capitalization clauses are cut; in practice cutting them costs nothing
  measurable, and no long-clause slice underperforms. The `max_length = 256` choice is validated.
- **The "fewer than 50 training examples" slice is too small to report.** It contains 11 test
  clauses spanning 3 classes and swings between 0.36 and 0.55 across seeds. It is replaced above
  by the bottom-40-class slice (n = 1,625), which is large enough to be meaningful.
- **Out-of-scope and French text is rejected well.** Mean confidence 0.38 and 0.34 against 0.97 on
  real clauses; only 2.4% of non-clause samples and 11% of French samples exceed 0.9 confidence.
  The model does not confidently classify things it should not.
- **Rare classes are overconfident.** The bottom-40 slice has 0.948 mean confidence at 0.780 accuracy —
  a 17-point confidence/accuracy gap, the worst calibration in any slice. This is the risk that
  matters in production: rare clauses are exactly where a lawyer needs the tool to say "I don't know".

### Abstention
Confidence threshold sweep on the test set (`results/audit/seed_*/abstention.csv`, seed 42):

| Threshold | Coverage (tagged automatically) | Accuracy on tagged | Accuracy on referred |
|---|---|---|---|
| 0.50 | 96.9% | 0.890 | 0.350 |
| 0.70 | 89.7% | 0.917 | 0.495 |
| 0.85 | 82.0% | 0.940 | 0.571 |
| 0.95 | 67.9% | 0.964 | 0.682 |

A threshold of 0.85 tags 82% of clauses automatically at 94.0% accuracy and sends 18% to a lawyer.
The referred population sits at 57% accuracy, confirming the threshold routes the right cases away.
Note the *stated* risk: 18% of a due-diligence corpus is still a large volume of manual review.

## Limitations and risks
- The training contracts use US-centric drafting. The model has not been tested on EU contracts.
- Clauses longer than 256 tokens are cut. Measured effect: none on aggregate accuracy (+0.019 on the
  truncated slice); very long clauses (>512 tokens, 1.8% of the data) were not evaluated separately.
- Rare clause types (bottom 40 by frequency) are unreliable (0.780 accuracy) and overconfident (0.948).
- Labels come from clause headings, so they are noisy. **Of the 50 most confident errors, 48 are
  label noise, overlapping labels or multi-topic clauses, and only 2 are genuine model errors.**
  A large part of the measured error ceiling is the dataset, not the model.
- Retrieval is BM25-based; it relies on lexical overlap and will miss paraphrases that share no words
  with the bank. Its weakness is concentrated in rare clause types (hit@5 = 0.917).
- Retrieval evaluation rests on 30 queries, which is too few for tight confidence intervals.

## Data protection (GDPR)
Client contracts can contain personal data. Run the model and the clause bank on the client's
own infrastructure or in an EU region. Do not use client data to retrain without written agreement.

## Reproducibility
- Code tag: `v1.0`
- Entry point: `scripts/run_all.sh`
- Dependencies: `requirements.lock.txt`, `requirements.txt`
- Seeds: 42 / 1 / 2
- Hardware: NVIDIA RTX 4070 Ti SUPER (16 GB), ~40 min per seed; also run on Google Colab T4
  (~13 min per epoch). Scores agree within the seed spread across both.
- Set `CLAUSE_NLP_ROOT` to redirect Colab Drive output paths to a local directory; the configs
  themselves are unchanged.
