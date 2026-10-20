# Experiment log

All hyperparameters chosen on validation. Test used once per final configuration.

| Date | Model | Setting | Val macro-F1 | Test macro-F1 | Test acc. | Notes |
|---|---|---|---|---|---|---|
| 15 Oct | TF-IDF + LogReg | C=10, fixed by hand | – | 0.825 | 0.877 | First run, C not tuned (superseded) |
| 15 Oct | TF-IDF + LogReg | C ∈ {1, 10, 30}, best C=30 | 0.8185 | 0.8267 (dedup 0.8230) | 0.8775 | Gains flatten after C=10 |
| 16 Oct | Legal-BERT | smoke test, 2k rows, 1 ep | 0.026 | – | – | Pipeline check, 0.38 s/step on T4 |
| 17 Oct | Legal-BERT | 5 ep, lr 3e-5, len 256, seed 42 | 0.8127 | – | – | Still improving at ep 5; below baseline on val |
| 17 Oct | Legal-BERT | 10 ep, lr 3e-5, len 256, seed 42 | 0.8325 | 0.8298 [0.814–0.840] (dedup 0.8262) | 0.8843 | Chosen config. Best at ep 10. ECE 0.084 || 19–20 Oct | Legal-BERT | 3 seeds (42/1/2), local RTX 4070 Ti SUPER, 10 ep | 0.8251 / 0.8240 / 0.8184 | 0.8273 / 0.8256 / 0.8233 | 0.8862 / 0.8841 / 0.8823 | Mean 0.8254 ± 0.0020 macro-F1, 0.8842 ± 0.0020 acc, ECE 0.083 ± 0.002 |

### Notes on the 3-seed rerun
- Rerun locally so all three seeds share hardware, library versions and data cache. Seed 42
  reproduces the Colab run (val 0.8251 vs 0.8325, test 0.8273 vs 0.8298) within the seed spread.
- **Conclusion changes:** the 3-seed test mean (0.8254) does not beat the TF-IDF baseline (0.8267).
  The validation-only edge (0.8325 vs 0.8185) did not survive on test. Accuracy gain is real
  (McNemar p = 0.012) but macro-F1 gain is not.
- Legal-BERT is much worse calibrated than the baseline: ECE 0.083 vs 0.021.
- Baseline ECE measured at 0.021 (previously uncomputed).
- Truncation audit: clauses > 256 tokens score 0.903 vs 0.882 for short clauses. The EDA's
  truncation concern is not borne out.
- Retrieval: BM25 hit@5 0.967 / MRR 0.856 beats embeddings (0.900 / 0.786) and hybrid (0.933 / 0.832).
