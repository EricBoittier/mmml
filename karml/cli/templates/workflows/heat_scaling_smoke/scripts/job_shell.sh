#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../../.." && pwd)"
N="${1:?N}"; REPEAT="${2:?repeat}"; DT_SLUG="${3:?dt_slug}"
cd "$REPO_ROOT"
export KARML_CKPT="${KARML_CKPT:?export KARML_CKPT}"
KARML="${KARML_BIN:-karml}"
OUT="$REPO_ROOT/artifacts/pycharmm_mlpot/dcm${N}_npt_x64_${REPEAT}/${DT_SLUG}"
mkdir -p "$OUT"
exec "$KARML" md-system \
  --setup pycharmm_full \
  --backend pycharmm \
  --composition "DCM:${N}" \
  --checkpoint "$KARML_CKPT" \
  --output-dir "$OUT" \
  --md-stages mini,heat \
  --box-size 180 \
  --ps-heat 10 \
  --temperature 220 \
  --no-echeck \
  --quiet
