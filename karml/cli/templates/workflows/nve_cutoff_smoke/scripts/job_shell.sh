#!/usr/bin/env bash
set -euo pipefail
LEG="${1:?leg name}"
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../../.." && pwd)"
cd "$REPO_ROOT"
export KARML_CKPT="${KARML_CKPT:?export KARML_CKPT}"
OUT="$WORKFLOW_ROOT/results/$LEG"
mkdir -p "$OUT"
exec karml md-system \
  --setup pbc_nve \
  --backend pycharmm \
  --composition "DCM:5" \
  --checkpoint "$KARML_CKPT" \
  --output-dir "$OUT" \
  --md-stages mini,nve \
  --ps 1.0 \
  --mm-switch-on 7.0 \
  --quiet
