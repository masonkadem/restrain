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
WANDB_PROJECT="bp-estimation-v2"
EXTRA="$*"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8    # prevent cp1252 errors with torchinfo/wandb on Windows
export WANDB_PROJECT             # so log_sweep_summary.py / eda.py use the same project

if [[ "${CONDA_DEFAULT_ENV:-}" != "bp" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate bp
fi

PYTHON="${PYTHON:-python}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Format per entry:  "run_name | model | channels | extra_flags"
# run_name controls the checkpoint/JSON/W&B name, so the same model+channels can
# appear multiple times with different flags (e.g. cosine vs plateau scheduler).
EXPERIMENTS=(
    # LGBM first — fast and sets the feature-importance baseline
    "lgbm_ppg_ecg               | lgbm        | ppg,ecg      |"
    "lgbm_ppg_ecg_resp          | lgbm        | ppg,ecg,resp |"
    # S4 cross-channel — best model so far: test BOTH schedulers head-to-head
    "s4_cross_ppg_ecg_cosine    | s4_cross    | ppg,ecg      | --scheduler cosine"
    "s4_cross_ppg_ecg_plateau   | s4_cross    | ppg,ecg      | --scheduler plateau"
    "s4_cross_ppg_ecg_resp      | s4_cross    | ppg,ecg,resp | --scheduler cosine"
    # Other deep models (cosine default)
    "dual_stream_ppg_ecg        | dual_stream | ppg,ecg      |"
    "tri_stream_ppg_ecg_resp    | tri_stream  | ppg,ecg,resp |"
    "noise_robust_ppg_ecg       | noise_robust| ppg,ecg      |"
    "noise_robust_ppg_ecg_resp  | noise_robust| ppg,ecg,resp |"
    "s4_ppg_ecg                 | s4          | ppg,ecg      |"
    "s4_ppg_ecg_resp            | s4          | ppg,ecg,resp |"
    "s4_ppg                     | s4          | ppg          |"
    "transformer_ppg            | transformer | ppg          |"
    "transformer_ppg_ecg_resp   | transformer | ppg,ecg,resp |"
)

# Returns 0 if a result for the CURRENT architecture version already exists.
# Keyed on arch_version (bumped in train.py whenever models change) so stale
# results from a previous architecture are re-run rather than skipped.
ARCH_VERSION="v2"
already_done() {
    local name="$1" model="$2"
    if [[ "$model" == "lgbm" ]]; then
        local json="${DATA_ROOT}/results/${name}_sbp_best.json"
    else
        local json="${DATA_ROOT}/results/${name}_best.json"
    fi
    [[ -f "$json" ]] && grep -q "\"arch_version\": \"${ARCH_VERSION}\"" "$json" 2>/dev/null
}

run_experiment() {
    local name="$1" model="$2" channels="$3" flags="$4"
    local log_file="${LOG_DIR}/${name}.log"

    echo "------------------------------------------------------------"
    echo "  START  ${name}  (model=${model} channels=${channels} ${flags})"
    echo "  log -> ${log_file}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "${DATA_ROOT}/train.py" \
        --model         "$model"          \
        --channels      "$channels"       \
        --run_name      "$name"           \
        --gpu_profile   "$GPU_PROFILE"    \
        --downsample    "$DOWNSAMPLE"     \
        --seed          "$SEED"           \
        --wandb_project "$WANDB_PROJECT"  \
        --data_root     "$DATA_ROOT"      \
        $flags $EXTRA \
        2>&1 | tee "$log_file" \
    && echo "  DONE   ${name}" \
    || echo "  FAILED ${name}  (see ${log_file})"
}

echo "============================================================"
echo "  BP estimation sweep  |  25 Hz  |  GPU profile: ${GPU_PROFILE}"
echo "============================================================"

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name model channels flags <<< "$entry"
    # trim surrounding whitespace from each field
    name="$(echo -e "${name}" | xargs)"
    model="$(echo -e "${model}" | xargs)"
    channels="$(echo -e "${channels}" | xargs)"
    flags="$(echo -e "${flags}" | xargs)"

    if already_done "$name" "$model"; then
        echo "  SKIP   ${name}  (v2 result exists)"
        continue
    fi

    run_experiment "$name" "$model" "$channels" "$flags"
done

echo ""
echo "============================================================"
echo "  Sweep complete. Logs in: ${LOG_DIR}/"
echo "============================================================"

# Post comparison table to W&B (sweep_summary run)
echo "Logging sweep summary to W&B..."
WANDB_PROJECT="$WANDB_PROJECT" "$PYTHON" "${DATA_ROOT}/log_sweep_summary.py" "$DATA_ROOT" \
    && echo "  sweep_summary run posted to W&B." \
    || echo "  sweep_summary failed (non-fatal)."
