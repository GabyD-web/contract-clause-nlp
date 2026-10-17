# Experiment log

All hyperparameters chosen on validation. Test used once per final configuration.

| Date | Model | Setting | Val macro-F1 | Test macro-F1 | Test acc. | Notes |
|---|---|---|---|---|---|---|
| 15 Oct | TF-IDF + LogReg | C=10, fixed by hand | – | 0.825 | 0.877 | First run, C not tuned (superseded) |
| 15 Oct | TF-IDF + LogReg | C ∈ {1, 10, 30}, best C=30 | 0.8185 | 0.8267 (dedup 0.8230) | 0.8775 | Gains flatten after C=10 |
| 16 Oct | Legal-BERT | smoke test, 2k rows, 1 ep | 0.026 | – | – | Pipeline check, 0.38 s/step on T4 |
| 17 Oct | Legal-BERT | 5 ep, lr 3e-5, len 256, seed 42 | 0.8127 | – | – | Still improving at ep 5; below baseline on val |
| 17 Oct | Legal-BERT | 10 ep, lr 3e-5, len 256, seed 42 | 0.8325 | 0.8298 [0.814–0.840] (dedup 0.8262) | 0.8843 | Chosen config. Best at ep 10. ECE 0.084 |