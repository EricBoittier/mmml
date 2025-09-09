#!/usr/bin/env bash

# Common configuration for MCTS demo runs

# Project root and PYTHONPATH
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Default run parameters (override in caller if desired)
N_SIMULATIONS=${N_SIMULATIONS:-2000}
TEMPERATURE=${TEMPERATURE:-0.25}
TARGET_TOTAL=${TARGET_TOTAL:-44}
TARGET_SPAN=${TARGET_SPAN:-0}
NEUTRALITY_LAMBDA=${NEUTRALITY_LAMBDA:-0.0}
DQ_MAX=${DQ_MAX:-0.0}
DR_MAX=${DR_MAX:-0.0}
REFINE_STEPS=${REFINE_STEPS:-0}
REFINE_LR=${REFINE_LR:-1e-4}
ALPHA_E2E=${ALPHA_E2E:-0.0}

# Helper to run demo with current configuration
run_demo() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  echo "Running MCTS demo... outputs -> $out_dir"
  python "$ROOT_DIR/scripts/run_mcts_demo.py" \
    --n-simulations "$N_SIMULATIONS" \
    --temperature "$TEMPERATURE" \
    --target-total "$TARGET_TOTAL" \
    --target-span "$TARGET_SPAN" \
    --neutrality-lambda "$NEUTRALITY_LAMBDA" \
    --dq-max "$DQ_MAX" \
    --dr-max "$DR_MAX" \
    --refine-steps "$REFINE_STEPS" \
    --refine-lr "$REFINE_LR" \
    --alpha-e2e "$ALPHA_E2E" \
    --verbose \
    --save-prefix "$out_dir/run"
}


