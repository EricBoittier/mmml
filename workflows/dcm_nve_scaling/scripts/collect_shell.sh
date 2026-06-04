#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_DIR="${1:?workflow_dir}"
OUT_CSV="${2:?out.csv}"
OUT_MD="${3:?out.md}"
PY="${MMML_PYTHON:-python3}"
if [[ -x "$(dirname "$WORKFLOW_DIR")/../../.venv/bin/python" ]]; then
  PY="$(cd "$(dirname "$WORKFLOW_DIR")/../.." && pwd)/.venv/bin/python"
fi
exec "$PY" "$WORKFLOW_DIR/scripts/collect_scaling.py" \
  --config "$WORKFLOW_DIR/config.yaml" \
  --csv "$OUT_CSV" \
  --md "$OUT_MD"
