#!/usr/bin/env bash
# Reproduce every number in the README and the model card.
#
#   ./scripts/run_all.sh            # full pipeline (~3h on an RTX 4070 Ti SUPER)
#   SKIP_TRAIN=1 ./scripts/run_all.sh   # reuse existing runs/, redo eval only
#
# Configs keep the Colab Drive paths; CLAUSE_NLP_ROOT redirects every output
# to a local directory so the same configs work on a workstation.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export CLAUSE_NLP_ROOT=${CLAUSE_NLP_ROOT:-runs}
mkdir -p models "$CLAUSE_NLP_ROOT" results logs   # models/ is gitignored since 872c202
export HF_HUB_DISABLE_PROGRESS_BARS=1
SEEDS=${SEEDS:-"42 1 2"}
CONFIG=${CONFIG:-configs/legalbert_e10.yaml}
MAX_LEN=${MAX_LEN:-256}

echo "== clause bank and classifier =="
if [ -z "${SKIP_TRAIN:-}" ]; then
  echo "-- baseline: TF-IDF + logistic regression (C tuned on validation)"
  $PY -u -m src.baseline

  for seed in $SEEDS; do
    echo "-- Legal-BERT, seed $seed"
    $PY -u -m src.train --config "$CONFIG" --seed "$seed"
  done
fi

echo "== evaluation =="
for seed in $SEEDS; do
  echo "-- test metrics, seed $seed"
  $PY -u -m src.evaluate --model_dir "$CLAUSE_NLP_ROOT/legalbert_e10_s$seed" \
      --out_dir "results/legalbert_e10_s$seed" --max_length "$MAX_LEN"
  echo "-- audit, seed $seed"
  $PY -u -m src.audit --model_dir "$CLAUSE_NLP_ROOT/legalbert_e10_s$seed" \
      --out_dir "results/audit/seed_$seed" --max_length "$MAX_LEN"
done

echo "-- retrieval (BM25, embeddings, hybrid) on the 30 lawyer queries"
$PY -u -m src.eval_retrieval --bank_split train --k 5 --out results/retrieval

echo "== summary =="
$PY - <<'PYEOF'
import glob, json, statistics as st
rows = [json.load(open(f)) for f in sorted(glob.glob("results/legalbert_e10_s*/test_metrics.json"))]
keys = ["accuracy", "macro_f1", "macro_f1_dedup", "ece"]
for k in keys:
    vals = [r[k] for r in rows]
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {k:16s} {mean:.4f} +/- {sd:.4f}  (n={len(vals)} seeds)")
PYEOF

echo
echo "done. baseline ECE needs the fitted vectoriser: (see notebooks/02_evaluation.ipynb)"
