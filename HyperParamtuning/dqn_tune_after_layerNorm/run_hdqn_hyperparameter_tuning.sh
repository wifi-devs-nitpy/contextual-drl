#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export JAX_CACHE_COMPILATION_DIR="./jax_cache"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"

# ============================================================
# HDQN Hyperparameter Tuning Runner
# ============================================================
#
# Script:
#   tune_hdqn_hyperparameters.py
#
# Default search:
#   1000 configurations
#   5 runs/configuration
#   10,000 steps/run
#
# EVERY LEVEL (L1-L4) gets its own:
#   optimizer
#   learning rate
#   replay buffer
#   batch size
#   replay steps
#   epsilon minimum
#   epsilon decay
#   AdamW weight decay
#
# The Python tuner checkpoints EACH RUN, not merely each
# configuration. Therefore a crash halfway through a
# configuration is resumable.
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/tune_hdqn_hyperparameters.py"

N_CONFIGS="${N_CONFIGS:-1000}"
N_RUNS="${N_RUNS:-5}"
N_STEPS="${N_STEPS:-6000}"

SEED="${SEED:-42}"
SEARCH_SEED="${SEARCH_SEED:-20261006}"

D_AP="${D_AP:-10}"
D_STA_1="${D_STA_1:-2}"
D_STA_2="${D_STA_2:-4}"

N_LINKS="${N_LINKS:-3}"
N_TX_POWER_LEVELS="${N_TX_POWER_LEVELS:-4}"

WINDOW_SIZE="${WINDOW_SIZE:-100}"
CONFIDENCE="${CONFIDENCE:-0.99}"
TARGET_THROUGHPUT="${TARGET_THROUGHPUT:-1500}"

OUTPUT_DIR="${OUTPUT_DIR:-./hdqn_hyperparameter_tuning}"
TOP_K="${TOP_K:-12}"

FORCE="${FORCE:-0}"
SHOW="${SHOW:-0}"

SCREEN_RUNS="${SCREEN_RUNS:-}"
SCREEN_STEPS="${SCREEN_STEPS:-}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: ${PYTHON_SCRIPT} not found."
    exit 1
fi

echo "============================================================"
echo " HDQN HYPERPARAMETER TUNING"
echo "============================================================"
echo "Configurations : ${N_CONFIGS}"
echo "Runs/config    : ${N_RUNS}"
echo "Steps/run      : ${N_STEPS}"
echo "Seed           : ${SEED}"
echo "Search seed    : ${SEARCH_SEED}"
echo "Scenario       : d_ap=${D_AP}, d_sta=${D_STA_1}->${D_STA_2}"
echo "Target         : ${TARGET_THROUGHPUT} Mbps"
echo "Output         : ${OUTPUT_DIR}"
echo "Force rerun    : ${FORCE}"
echo "============================================================"
echo

ARGS=(
    --n-configs "${N_CONFIGS}"
    --n-runs "${N_RUNS}"
    --n-steps "${N_STEPS}"
    --seed "${SEED}"
    --search-seed "${SEARCH_SEED}"
    --d-ap "${D_AP}"
    --d-sta-1 "${D_STA_1}"
    --d-sta-2 "${D_STA_2}"
    --n-links "${N_LINKS}"
    --n-tx-power-levels "${N_TX_POWER_LEVELS}"
    --window-size "${WINDOW_SIZE}"
    --confidence "${CONFIDENCE}"
    --target-throughput "${TARGET_THROUGHPUT}"
    --output-dir "${OUTPUT_DIR}"
    --top-k "${TOP_K}"
)

if [[ "${FORCE}" == "1" ]]; then
    ARGS+=(--force)
fi

if [[ "${SHOW}" == "1" ]]; then
    ARGS+=(--show)
fi

if [[ -n "${SCREEN_RUNS}" ]]; then
    ARGS+=(--screen-runs "${SCREEN_RUNS}")
fi

if [[ -n "${SCREEN_STEPS}" ]]; then
    ARGS+=(--screen-steps "${SCREEN_STEPS}")
fi

python "${PYTHON_SCRIPT}" "${ARGS[@]}"

echo
echo "============================================================"
echo " TUNING FINISHED"
echo "============================================================"
echo "Summary : ${OUTPUT_DIR}/summary.csv"
echo "Configs : ${OUTPUT_DIR}/configs.csv"
echo "Cache   : ${OUTPUT_DIR}/cache/"
echo "Plots   : ${OUTPUT_DIR}/plots/"
echo "============================================================"
