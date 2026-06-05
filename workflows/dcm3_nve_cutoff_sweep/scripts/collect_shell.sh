#!/usr/bin/env bash
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"
PY="${MMML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi
exec "$PY" "$WORKFLOW_ROOT/scripts/collect_sweep.py" \
  --config "$WORKFLOW_ROOT/config.yaml" \
  --csv "$WORKFLOW_ROOT/results/cutoff_sweep_summary.csv" \
  --md "$WORKFLOW_ROOT/results/cutoff_sweep_report.md"
