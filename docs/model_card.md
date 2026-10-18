# Model card: contract clause classifier and clause-bank retrieval

## Model details
- **Classifier:** Legal-BERT (`nlpaueb/legal-bert-base-uncased`) fine-tuned on LEDGAR, 100 clause types,
  10 epochs, max 256 tokens, 3 seeds.
- **Retrieval:** sentence-transformer embeddings + FAISS, combined with BM25 keyword search.
- **Baseline:** TF-IDF + logistic regression.
- **Version:** v1.0, October 2026.
- **Authors:** Gaby, Theodore, Leopold.

## Intended use
The model helps qualified lawyers during due diligence on English-language commercial contracts.
It suggests the type of each clause and shows similar clauses from the clause bank.

A lawyer reviews every output. Clauses below the confidence threshold are sent to a lawyer.

## Out-of-scope uses: the model must NOT be used for
- Legal advice, or deciding whether a clause is valid, enforceable or risky
- Contracts not written in English (French accuracy: [TBD] vs [TBD] in English)
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

### Classification (test set, 10,000 clauses)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.878 | 0.827 | 0.874 |
| Legal-BERT (mean ± std over 3 seeds) | [TBD] ± [TBD] | [TBD] ± [TBD] | [TBD] ± [TBD] |

- Legal-BERT is better than the keyword baseline on accuracy, but not significantly better on macro-F1 ([TBD: test used, p-value]).
- It is weaker on rare clause types. Most errors are between near-synonym labels.
  - The baseline's lowest F1 scores are on near-synonym labels: Applicable Laws (0.30, confused with Governing Laws), Jurisdictions (0.38), Venues (0.49), Miscellaneous (0.59), Agreements (0.59) and Modifications (0.60, confused with Amendments).
  - Legal-BERT on these classes: [TBD].
- Confidence threshold [TBD]: [TBD]% of clauses are tagged automatically at [TBD]% accuracy. The remaining [TBD]% are sent to a lawyer.

Per-class results: `results/baseline/per_class.csv`, `results/legalbert/per_class.csv`.

### Retrieval (30 lawyer queries, `data/queries.csv`)
The 30 queries cover 10 frequent, 10 mid-frequency and 10 rare clause types. 5 of them contain numbers.

| Method | Hit@5 (correct label) | MRR | must_contain found in top 5 |
|---|---|---|---|
| BM25 only | [TBD] | [TBD] | [TBD] |
| Embeddings + FAISS only | [TBD] | [TBD] | [TBD] |
| Hybrid (BM25 + embeddings) | [TBD] | [TBD] | [TBD] |

30 queries is a small test set. One query is worth 3.3 points of Hit@5.

## Audit findings
[TBD: table from results/audit/summary.csv]

| Slice | n | Accuracy | Macro-F1 | Gap vs overall |
|---|---|---|---|---|
| Frequent clause types | [TBD] | [TBD] | [TBD] | [TBD] |
| Rare clause types (fewer than 50 training examples) | [TBD] | [TBD] | [TBD] | [TBD] |
| Clauses longer than 256 tokens | [TBD] | [TBD] | [TBD] | [TBD] |
| French clauses | [TBD] | [TBD] | [TBD] | [TBD] |
| Non-clause text (expected: low confidence) | [TBD] | mean confidence [TBD] | n/a | n/a |

## Limitations and risks
- The training contracts use US-centric drafting. The model has not been tested on EU contracts.
- Clauses longer than 256 tokens are cut: [TBD effect].
- The model is overconfident on [TBD: French / non-clause text].
- Rare clause types (fewer than 50 training examples) are unreliable.
- Labels come from clause headings, so they are noisy. Part of the measured error is label noise, not model error.

## Data protection (GDPR)
Client contracts can contain personal data. Run the model and the clause bank on the client's
own infrastructure or in an EU region. Do not use client data to retrain without written agreement.

## Reproducibility
- Code tag: `v1.0`
- Entry point: `scripts/run_all.sh`
- Dependencies: `requirements.lock.txt`
- Seeds: 42 / 1 / 2
- Hardware: Google Colab T4 GPU
