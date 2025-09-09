#!/usr/bin/env bash
set -euo pipefail

# MCTS with refinement steps on selected q (no NN deltas)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# refinement overrides
NEUTRALITY_LAMBDA=0.0
DQ_MAX=0.0
DR_MAX=0.0
REFINE_STEPS=${REFINE_STEPS:-10}
REFINE_LR=${REFINE_LR:-5e-4}
OUT_DIR="$ROOT_DIR/outputs/mcts_refine/$(date +%Y%m%d_%H%M%S)"
run_demo "$OUT_DIR"

echo "Done."


