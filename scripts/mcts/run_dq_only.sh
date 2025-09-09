#!/usr/bin/env bash
set -euo pipefail

# MCTS with small charge corrections only (no displacement), mild neutrality penalty

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# dq-only overrides
NEUTRALITY_LAMBDA=1e-4
DQ_MAX=0.05
DR_MAX=0.0
REFINE_STEPS=0
OUT_DIR="$ROOT_DIR/outputs/mcts_dq_only/$(date +%Y%m%d_%H%M%S)"
run_demo "$OUT_DIR"

echo "Done."


