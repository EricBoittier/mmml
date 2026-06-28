#!/usr/bin/env bash
set -euo pipefail
LEG="${1:?leg name}"
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../../.." && pwd)"
cd "$REPO_ROOT"
export MMML_CKPT="${MMML_CKPT:?export MMML_CKPT}"
OUT="$WORKFLOW_ROOT/results/$LEG"
mkdir -p "$OUT"
exec mmml md-system \
  --setup pbc_nve \
  --backend pycharmm \
  --composition "DCM:5" \
  --checkpoint "$MMML_CKPT" \
  --output-dir "$OUT" \
  --md-stages mini,nve \
  --ps 1.0 \
  --mm-switch-on 7.0 \
  --quiet
