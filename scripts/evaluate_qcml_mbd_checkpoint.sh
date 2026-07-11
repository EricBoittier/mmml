#!/usr/bin/env bash
set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
PY="${MMML_PYTHON:-$ROOT/.venv/bin/python}"
CACHE="${CACHE:-$HOME/orbax_cache/qcml_mbd}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Usage: $0 /path/to/epoch-XXXX [output_dir]" >&2
  echo "Or set CHECKPOINT=/path/to/epoch-XXXX" >&2
  exit 2
fi

OUTPUT_DIR="${2:-${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/eval_$(basename "$CHECKPOINT")_test}}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BUCKET_WIDTH="${BUCKET_WIDTH:-8}"
MAX_ATOMS="${MAX_ATOMS:-32}"
MAX_STRUCTURES="${MAX_STRUCTURES:-}"

cd "$ROOT"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

args=(
  --cache "$CACHE"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --bucket-width "$BUCKET_WIDTH"
  --max-atoms "$MAX_ATOMS"
)

if [[ -n "$MAX_STRUCTURES" ]]; then
  args+=(--max-structures "$MAX_STRUCTURES")
fi

"$PY" scripts/analyze_qcml_mbd.py "${args[@]}"
