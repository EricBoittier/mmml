#!/usr/bin/env bash
set -euo pipefail

# Baseline pure MCTS (no NN corrections, no neutrality penalty)

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# baseline overrides
NEUTRALITY_LAMBDA=0.0
DQ_MAX=0.0
DR_MAX=0.0
REFINE_STEPS=0
OUT_DIR="$ROOT_DIR/outputs/mcts_baseline/$(date +%Y%m%d_%H%M%S)"
run_demo "$OUT_DIR"

echo "Done."


