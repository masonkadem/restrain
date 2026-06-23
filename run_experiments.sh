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

set -uo pipefail   # no -e: a single failed run logs and continues

# ── Defaults (can be overridden on the command line) ─────────────────────────
GPU_PROFILE="3080"   # fast | 3080 | h100
CUDA_DEVICE="0"      # which GPU index to use (CUDA_VISIBLE_DEVICES)
SEED=42
DATA_ROOT="$(cd "$(dirname "$0")" && pwd)"
WANDB_PROJECT="bp-estimation"

# Forward any extra flags to all runs (e.g. --epochs 50 --lr 1e-4)
EXTRA="$*"

# Help PyTorch handle fragmented VRAM between sequential runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "bp" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bp
fi

# ── Channel sets ─────────────────────────────────────────────────────────────
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

# ── Per-model batch size overrides (dual/tri have 2-3× more activation memory)
declare -A BATCH_OVERRIDE
BATCH_OVERRIDE[dual_stream]=8
BATCH_OVERRIDE[tri_stream]=8

# ── Utilities ─────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Returns 0 if this run already has a real (non-smoke) result saved.
# Uses grep so Git Bash paths work without needing Python path conversion.
already_done() {
    local model="$1"
    local channels="$2"
    local name_tag="${model}_${channels//,/_}"   # matches train.py run_name

    if [[ "$model" == "lgbm" ]]; then
        local json="${DATA_ROOT}/results/${name_tag}_sbp_best.json"
        [[ -f "$json" ]] && grep -q '"n_estimators": 2000' "$json" 2>/dev/null
    else
        local json="${DATA_ROOT}/results/${name_tag}_best.json"
        [[ -f "$json" ]] && grep -q '"epochs": 100' "$json" 2>/dev/null
    fi
}

run_experiment() {
    local model="$1"
    local channels="$2"
    local tag="${model}__${channels//,/_}"
    local log_file="${LOG_DIR}/${tag}.log"
    local batch_flag=""

    if [[ -n "${BATCH_OVERRIDE[$model]+x}" ]]; then
        batch_flag="--batch_size ${BATCH_OVERRIDE[$model]}"
    fi

    echo "------------------------------------------------------------"
    echo "  START  model=${model}  channels=${channels}${batch_flag:+  (batch override: ${BATCH_OVERRIDE[$model]})}"
    echo "  log -> ${log_file}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "${DATA_ROOT}/train.py" \
        --model        "$model"          \
        --channels     "$channels"       \
        --gpu_profile  "$GPU_PROFILE"    \
        --seed         "$SEED"           \
        --wandb_project "$WANDB_PROJECT" \
        --data_root    "$DATA_ROOT"      \
        $batch_flag                      \
        $EXTRA \
        2>&1 | tee "$log_file" \
    && echo "  DONE   model=${model}  channels=${channels}" \
    || echo "  FAILED model=${model}  channels=${channels}  (see ${log_file})"
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

        # Skip runs that already have a real (non-smoke) result
        if already_done "$model" "$channels"; then
            echo "  SKIP   model=${model}  channels=${channels}  (already completed)"
            continue
        fi

        run_experiment "$model" "$channels"
    done
done

echo ""
echo "============================================================"
echo "  All experiments complete. Logs in: ${LOG_DIR}/"
echo "============================================================"
