# Experiment log

All hyperparameters chosen on validation. Test used once per final configuration.

| Date | Model | Setting | Hardware | Val macro-F1 | Test macro-F1 | Test acc. | Notes |
|---|---|---|---|---|---|---|---|
| 15 Oct | TF-IDF + LogReg | C=10, fixed by hand | laptop CPU | – | 0.825 | 0.877 | First run, C not tuned (superseded) |
| 15 Oct | TF-IDF + LogReg | C ∈ {1, 10, 30}, best C=30 | laptop CPU | 0.8185 | 0.8267 (dedup 0.8230) | 0.8775 | Gains flatten after C=10. **Reference baseline.** ECE 0.020 |
| 16 Oct | Legal-BERT | smoke test, 2k rows, 1 ep | Colab T4 | 0.026 | – | – | Pipeline check, 0.38 s/step |
| 17 Oct | Legal-BERT | 5 ep, lr 3e-5, len 256, seed 42 | Colab T4 | 0.8127 | – | – | Still improving at ep 5; below baseline on val |
| 17 Oct | Legal-BERT | 10 ep, lr 3e-5, len 256, seed 42 | Colab T4 | 0.8325 | 0.8298 [0.814–0.840] (dedup 0.8262) | 0.8843 | **Chosen config.** Best at ep 10. ECE 0.084 |
| 18 Oct | Legal-BERT | 10 ep, seed 1 | Colab T4 | – | – | – | Interrupted at ep 9 (GPU quota). Discarded, rerun on Kaggle |
| 19 Oct | Legal-BERT | 10 ep, seed 1 | Kaggle T4 | 0.8248 | 0.8318 [0.818–0.841] (dedup 0.8284) | 0.8873 | ECE 0.083 |
| 19 Oct | Legal-BERT | 10 ep, seed 2 | Kaggle T4 | 0.8230 (ep 9) | 0.8240 [0.810–0.833] (dedup 0.8204) | 0.8827 | ECE 0.085 |
| 19–20 Oct | Legal-BERT | 10 ep, seed 42 | local RTX 4070 Ti SUPER | 0.8251 | 0.8273 [0.813–0.836] (dedup 0.8238) | 0.8862 | ECE 0.081. Used by the audit |
| 19–20 Oct | Legal-BERT | 10 ep, seed 1 | local RTX 4070 Ti SUPER | 0.8240 (ep 9) | 0.8256 [0.811–0.835] (dedup 0.8222) | 0.8841 | ECE 0.083. Used by the audit |
| 19–20 Oct | Legal-BERT | 10 ep, seed 2 | local RTX 4070 Ti SUPER | 0.8184 (ep 9) | 0.8233 [0.809–0.832] (dedup 0.8198) | 0.8823 | ECE 0.085. Used by the audit |
| 20 Oct | TF-IDF + LogReg | rerun, C=30 | local workstation | 0.8188 | 0.8266 (dedup 0.8229) | 0.8767 | Reproduces the reference baseline within 0.1 pt |
| 20 Oct | Legal-BERT + temperature scaling | T fitted on val: 1.45 (s1), 1.43 (s2) | laptop CPU | 0.8248 / 0.8230 | 0.8318 / 0.8240 (unchanged) | 0.8873 / 0.8827 | ECE 0.083 → 0.024 (s1), 0.085 → 0.028 (s2) |
| 20 Oct | **Ensemble: cal. Legal-BERT + TF-IDF, 50/50** | seeds 1 / 2, no fitted weights | laptop CPU | 0.8325 / 0.8365 | **0.844 / 0.836** | **0.895 / 0.891** | ECE 0.017 / 0.013. +1.2 pt macro-F1 over Legal-BERT (95% CI excludes 0; McNemar p < 1e-7). Post-hoc idea, confirmed on validation. ~77% auto-tagged at ~97% accuracy |

Local-GPU runs: `results/local_gpu/legalbert_e10_s*/`. Colab/Kaggle runs: `results/legalbert_e10_s*/`.

## Summary of the six Legal-BERT runs

| Run set | Accuracy | Macro-F1 | Macro-F1 dedup | ECE |
|---|---|---|---|---|
| Colab/Kaggle T4 (3 seeds) | 0.8848 ± 0.0023 | 0.8286 ± 0.0040 | 0.8250 ± 0.0041 | 0.084 ± 0.001 |
| Local RTX 4070 Ti SUPER (3 seeds) | 0.8842 ± 0.0020 | 0.8254 ± 0.0020 | 0.8219 ± 0.0020 | 0.083 ± 0.002 |
| **All 6 runs** | **0.8845 ± 0.0019** | **0.8270 ± 0.0033** | 0.8235 ± 0.0033 | 0.083 ± 0.001 |
| Baseline (reference) | 0.8775 | 0.8267 | 0.8230 | 0.020 |

## Notes
- **Conclusion.** Legal-BERT is more accurate than TF-IDF in every one of the six runs (+0.7 pt;
  McNemar p = 0.012 on seed 42) but not better on macro-F1: the six-run mean equals the baseline,
  and single runs fall on both sides of it. The validation edge of the first run (0.8325 vs 0.8185)
  was partly seed luck: the six-run validation mean is 0.8246 ± 0.0046.
- **Same seed, different hardware** gives up to 0.6 pt different test macro-F1 (seed 1: 0.8318 vs
  0.8256), because GPU computations are not bit-for-bit deterministic. Single-run comparisons are
  therefore not meaningful at this scale; headline numbers are means over six runs.
- Legal-BERT is much worse calibrated than the baseline (ECE 0.083 vs 0.020).
- Truncation audit (local runs): clauses > 256 tokens score 0.903 vs 0.882 for short clauses. The
  EDA's truncation concern is not borne out.
- Retrieval (30 lawyer queries): BM25 hit@5 0.967 / MRR 0.856 beats embeddings (0.900 / 0.786) and
  hybrid (0.933 / 0.832).
- **Calibration and ensemble (seeds 1 and 2, `notebooks/02_evaluation.ipynb`).** Temperature scaling
  (T ≈ 1.44, fitted on validation) brings Legal-BERT's ECE from 0.083 to 0.024–0.028. At equal
  coverage, calibrated Legal-BERT ranks its confidence no better than TF-IDF. Averaging the two
  models' probabilities is significantly better than Legal-BERT alone (macro-F1 +1.2 pt) and tags
  about 77% of clauses at about 97% accuracy (thresholds chosen on validation). Files:
  `results/abstention.json`, `results/ensemble.json`, `results/ensemble_summary.csv`.
