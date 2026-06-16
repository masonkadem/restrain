#!/usr/bin/env bash
# run_experiments.sh — sweep all model × channel permutations
#
# Usage:
#   bash run_experiments.sh
#   bash run_experiments.sh --epochs 40      # override epochs for quick tests
#   CUDA_DEVICE=1 bash run_experiments.sh    # target a specific GPU index
#
# Outputs:
#   Each run saves a <model>_<channels>_best.pt checkpoint.
#   All runs are logged to W&B project "bp-estimation".

set -euo pipefail

# ── Defaults (can be overridden on the command line) ─────────────────────────
GPU_PROFILE="3080"   # fast | 3080 | h100
CUDA_DEVICE="0"      # which GPU index to use (CUDA_VISIBLE_DEVICES)
SEED=42
DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
WANDB_PROJECT="bp-estimation"

# Forward any extra flags to all runs (e.g. --epochs 50 --lr 1e-4)
EXTRA="$*"

# Activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "bp" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bp
fi

# ── Channel sets ─────────────────────────────────────────────────────────────
# Each entry is a comma-separated list of channel names.
CHANNEL_SETS=(
    "ppg"
    "ppg,ecg"
    "ppg,resp"
    "ppg,ecg,resp"
)

# ── Models ───────────────────────────────────────────────────────────────────
# dual_stream only runs with ppg,ecg — enforced in the loop below.
# tri_stream only runs with ppg,ecg,resp — enforced in the loop below.
MODELS=(transformer dual_stream tri_stream s4 lgbm)

# ── Utilities ─────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"

run_experiment() {
    local model="$1"
    local channels="$2"
    local tag="${model}__${channels//,/_}"
    local log_file="${LOG_DIR}/${tag}.log"

    echo "------------------------------------------------------------"
    echo "  START  model=${model}  channels=${channels}"
    echo "  log -> ${log_file}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "${DATA_ROOT}/train.py" \
        --model        "$model"          \
        --channels     "$channels"       \
        --gpu_profile  "$GPU_PROFILE"    \
        --seed         "$SEED"           \
        --wandb_project "$WANDB_PROJECT" \
        --data_root    "$DATA_ROOT"      \
        $EXTRA \
        2>&1 | tee "$log_file"

    echo "  DONE   model=${model}  channels=${channels}"
}

# ── Main sweep ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  BP estimation experiment sweep"
echo "  GPU profile: ${GPU_PROFILE}"
echo "============================================================"

for model in "${MODELS[@]}"; do
    for channels in "${CHANNEL_SETS[@]}"; do

        # dual_stream requires exactly ppg,ecg
        if [[ "$model" == "dual_stream" && "$channels" != "ppg,ecg" ]]; then
            echo "  SKIP   model=dual_stream  channels=${channels}  (requires ppg,ecg)"
            continue
        fi

        # tri_stream requires exactly ppg,ecg,resp
        if [[ "$model" == "tri_stream" && "$channels" != "ppg,ecg,resp" ]]; then
            echo "  SKIP   model=tri_stream  channels=${channels}  (requires ppg,ecg,resp)"
            continue
        fi

        run_experiment "$model" "$channels"
    done
done

echo ""
echo "============================================================"
echo "  All experiments complete. Logs in: ${LOG_DIR}/"
echo "============================================================"
