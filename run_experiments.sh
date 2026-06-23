#!/usr/bin/env bash
# run_experiments.sh — 25 Hz sweep focused on dual/tri/S4 for BP estimation
#
# Usage:
#   bash run_experiments.sh                    # full sweep
#   CUDA_DEVICE=1 bash run_experiments.sh      # target GPU 1
#   bash run_experiments.sh --epochs 10        # quick test
#
# Sampling: loaded at 125 Hz → decimated to 25 Hz (--downsample 5).
# At 25 Hz seq_len=375 — batch=32 works for all models including dual/tri stream.

set -uo pipefail   # no -e: single failure logs and continues

GPU_PROFILE="3080"
DOWNSAMPLE=5        # 125 → 25 Hz
CUDA_DEVICE="0"
SEED=42
DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
WANDB_PROJECT="bp-estimation"
EXTRA="$*"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${CONDA_DEFAULT_ENV:-}" != "bp" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bp
fi

PYTHON="${PYTHON:-python}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Ordered by priority: dual/tri stream and S4 first, then transformer/lgbm baselines.
# Format: "model  channels"
EXPERIMENTS=(
    "dual_stream  ppg,ecg"
    "tri_stream   ppg,ecg,resp"
    "s4           ppg,ecg"
    "s4           ppg,ecg,resp"
    "s4           ppg"
    "transformer  ppg"
    "transformer  ppg,ecg,resp"
    "lgbm         ppg,ecg"
    "lgbm         ppg,ecg,resp"
)

# Returns 0 if a 25 Hz result (downsample=5) already exists for this run.
already_done() {
    local model="$1" channels="$2"
    local name_tag="${model}_${channels//,/_}"

    if [[ "$model" == "lgbm" ]]; then
        local json="${DATA_ROOT}/results/${name_tag}_sbp_best.json"
        [[ -f "$json" ]] && grep -q '"downsample": 5' "$json" 2>/dev/null
    else
        local json="${DATA_ROOT}/results/${name_tag}_best.json"
        [[ -f "$json" ]] && grep -q '"downsample": 5' "$json" 2>/dev/null
    fi
}

run_experiment() {
    local model="$1" channels="$2"
    local tag="${model}__${channels//,/_}"
    local log_file="${LOG_DIR}/${tag}.log"

    echo "------------------------------------------------------------"
    echo "  START  model=${model}  channels=${channels}  (25 Hz, batch=32)"
    echo "  log -> ${log_file}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "${DATA_ROOT}/train.py" \
        --model         "$model"          \
        --channels      "$channels"       \
        --gpu_profile   "$GPU_PROFILE"    \
        --downsample    "$DOWNSAMPLE"     \
        --seed          "$SEED"           \
        --wandb_project "$WANDB_PROJECT"  \
        --data_root     "$DATA_ROOT"      \
        $EXTRA \
        2>&1 | tee "$log_file" \
    && echo "  DONE   model=${model}  channels=${channels}" \
    || echo "  FAILED model=${model}  channels=${channels}  (see ${log_file})"
}

echo "============================================================"
echo "  BP estimation sweep  |  25 Hz  |  GPU profile: ${GPU_PROFILE}"
echo "============================================================"

for entry in "${EXPERIMENTS[@]}"; do
    read -r model channels <<< "$entry"

    if already_done "$model" "$channels"; then
        echo "  SKIP   model=${model}  channels=${channels}  (25 Hz result exists)"
        continue
    fi

    run_experiment "$model" "$channels"
done

echo ""
echo "============================================================"
echo "  Sweep complete. Logs in: ${LOG_DIR}/"
echo "============================================================"
