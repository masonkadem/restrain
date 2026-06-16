#!/usr/bin/env bash
# smoke_test.sh — quick sanity check for all models and channel combos.
#
# Runs 3 epochs with 400 train / 100 test samples via train.py --smoke.
# Expected duration: ~2-4 min on GPU.
#
# Usage:
#   bash smoke_test.sh
#   bash smoke_test.sh --gpu_profile fast    # override profile

set -euo pipefail

DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
EXTRA="${*}"   # forward any extra flags to train.py

# Activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "bp" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bp
fi

# Use the "fast" profile by default for smoke tests (8s window, small model)
SMOKE_FLAGS="--smoke --epochs 3 --gpu_profile fast --data_root ${DATA_ROOT} ${EXTRA}"

PASS=0; FAIL=0; SKIP=0
RESULTS=()

run_smoke() {
    local model="$1"; local channels="$2"
    local tag="${model}[${channels}]"
    echo -n "  ${tag}  ... "
    if output=$("${PYTHON}" "${DATA_ROOT}/train.py" --model "${model}" \
                    --channels "${channels}" ${SMOKE_FLAGS} 2>&1); then
        echo "PASS"
        RESULTS+=("PASS  ${tag}")
        PASS=$((PASS+1))
    else
        echo "FAIL"
        echo "    --- stderr ---"
        echo "${output}" | tail -5 | sed 's/^/    /'
        RESULTS+=("FAIL  ${tag}")
        FAIL=$((FAIL+1))
    fi
}

echo "============================================================"
echo "  BP Estimation — Smoke Test  (3 epochs, 400 samples)"
echo "============================================================"

# transformer: all channel subsets
run_smoke transformer   "ppg"
run_smoke transformer   "ppg,ecg"
run_smoke transformer   "ppg,resp"
run_smoke transformer   "ppg,ecg,resp"

# dual_stream: only ppg,ecg
run_smoke dual_stream   "ppg,ecg"

# tri_stream: only ppg,ecg,resp
run_smoke tri_stream    "ppg,ecg,resp"

# s4: all channel subsets
run_smoke s4            "ppg"
run_smoke s4            "ppg,ecg"
run_smoke s4            "ppg,resp"
run_smoke s4            "ppg,ecg,resp"

# lgbm: all channel subsets
run_smoke lgbm          "ppg"
run_smoke lgbm          "ppg,ecg"
run_smoke lgbm          "ppg,resp"
run_smoke lgbm          "ppg,ecg,resp"

echo ""
echo "============================================================"
printf "  Results: %d PASS  /  %d FAIL  /  %d total\n" $PASS $FAIL $((PASS+FAIL))
echo "============================================================"
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo ""

if [ $FAIL -gt 0 ]; then
    echo "⚠  Some tests failed — fix before running full experiments."
    exit 1
else
    echo "✓  All smoke tests passed.  Run: bash run_experiments.sh"
    exit 0
fi
