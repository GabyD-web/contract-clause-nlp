#!/usr/bin/env bash
# Quickstart in one command, on Linux/macOS, WSL and Windows.
#
#   ./scripts/quickstart.sh              # full quickstart, ~30 min (baseline on CPU)
#   ./scripts/quickstart.sh --skip-baseline   # environment + tests only, ~2 min
#   BASELINE_OUT=results/repro ./scripts/quickstart.sh
#
# What it does, matching the README quickstart:
#   1. creates .venv (skipped if one already exists)
#   2. activates it, using the right path for the OS
#   3. installs requirements.txt
#   4. runs the test suite
#   5. runs the TF-IDF baseline and prints its test macro-F1
#
# The baseline normally writes to results/baseline/, which holds the reference
# run that notebooks/02_evaluation.ipynb depends on. Set BASELINE_OUT to write
# somewhere else (results/repro is a good choice) and leave the reference intact.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
SKIP_BASELINE=0
for arg in "$@"; do
  case "$arg" in
    --skip-baseline) SKIP_BASELINE=1 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- platform
case "$(uname -s)" in
  Linux*)
    if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then PLATFORM="WSL"; else PLATFORM="Linux"; fi ;;
  Darwin*) PLATFORM="macOS" ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM="Windows (Git Bash)" ;;
  *) PLATFORM="unknown: $(uname -s)" ;;
esac
echo "platform : $PLATFORM"

# Pick the interpreter. On Windows, python3 is often the Microsoft Store stub,
# so prefer python/launcher order there and the reverse elsewhere.
find_python() {
  local candidates
  case "$PLATFORM" in
    Windows*) candidates=(python python3 py) ;;
    *)        candidates=(python3 python) ;;
  esac
  for c in "${candidates[@]}"; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}
PYTHON=$(find_python) || { echo "ERROR: no Python >= 3.9 found. Install Python and re-run." >&2; exit 1; }
echo "python   : $PYTHON ($("$PYTHON" -V 2>&1))"

# ---------------------------------------------------------------- venv
if [ -d .venv ]; then
  echo "venv     : reusing existing .venv"
else
  echo "venv     : creating"
  "$PYTHON" -m venv .venv
fi

case "$PLATFORM" in
  Windows*) VENV_BIN=".venv/Scripts" ;;
  *)        VENV_BIN=".venv/bin" ;;
esac
[ -x "$VENV_BIN/python" ] || [ -x "$VENV_BIN/python.exe" ] || {
  echo "ERROR: interpreter not found in $VENV_BIN after creating the venv." >&2
  echo "       On Windows, try running this from Git Bash; on Linux install python3-venv." >&2
  exit 1
}
# Call the venv interpreter directly: avoids relying on `source activate`,
# which behaves differently in Git Bash, WSL and POSIX shells.
VPY="$ROOT/$VENV_BIN/python"
[ -x "$VPY" ] || VPY="$ROOT/$VENV_BIN/python.exe"
echo "venv bin : $VENV_BIN"
echo "interp   : $VPY"

# ---------------------------------------------------------------- install
echo
echo "== installing requirements.txt =="
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -r requirements.txt

# Belt and braces: src/baseline.py imports joblib. It is listed in
# requirements.txt, but environments that skip that file (or a stale checkout)
# would otherwise fail at the very end of a 30-minute run.
"$VPY" -c "import joblib" 2>/dev/null || {
  echo "note: joblib not importable; installing it (src/baseline.py imports it)"
  "$VPY" -m pip install joblib
}

# ---------------------------------------------------------------- tests
echo
echo "== tests =="
"$VPY" -m pytest -q

if [ "$SKIP_BASELINE" = "1" ]; then
  echo
  echo "done (baseline skipped)."
  exit 0
fi

# ---------------------------------------------------------------- baseline
OUT=${BASELINE_OUT:-results/baseline}
if [ "$OUT" = "results/baseline" ]; then
  echo
  echo "WARNING: the baseline writes to results/baseline/, which holds the reference run"
  echo "         used by notebooks/02_evaluation.ipynb. It will be overwritten."
  echo "         Re-run with BASELINE_OUT=results/repro to keep the reference intact,"
  echo "         or press Ctrl-C now."
  sleep 5
fi
mkdir -p "$OUT" models

echo
echo "== baseline: TF-IDF + logistic regression (CPU, ~30 min) =="
echo "   writing to $OUT"
"$VPY" -m src.baseline

# src/baseline.py hardcodes results/baseline/ and models/, so if a different
# output was asked for, move the three artefacts there.
if [ "$OUT" != "results/baseline" ]; then
  mkdir -p "$OUT"
  for f in test_metrics.json test_pred.npy per_class.csv; do
    [ -f "results/baseline/$f" ] && cp "results/baseline/$f" "$OUT/$f"
  done
  echo "   copied metrics to $OUT (results/baseline/ was also written by the script)"
fi

echo
echo "== result =="
"$VPY" - "$OUT" <<'PYEOF'
import json, sys
path = sys.argv[1].rstrip("/") + "/test_metrics.json"
d = json.load(open(path))
t = d["test"]
print(f"  best C          : {d['best_C']}")
print(f"  test accuracy   : {t['accuracy']:.4f}")
print(f"  test macro-F1   : {t['macro_f1']:.4f}   (reference run: 0.8267)")
print(f"  test micro-F1   : {t['micro_f1']:.4f}")
PYEOF

echo
echo "done. Reference values: accuracy 0.8775, macro-F1 0.8267 (C=30)."
echo "A rerun on a different machine typically lands within 0.1 pt."
