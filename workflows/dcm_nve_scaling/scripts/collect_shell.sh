#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_DIR="${1:?workflow_dir}"
OUT_CSV="${2:?out.csv}"
OUT_MD="${3:?out.md}"
REPO_ROOT="$(cd "$(dirname "$WORKFLOW_DIR")/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"
exec "$PY" "$WORKFLOW_DIR/scripts/collect_scaling.py" \
  --config "$WORKFLOW_DIR/config.yaml" \
  --csv "$OUT_CSV" \
  --md "$OUT_MD"
