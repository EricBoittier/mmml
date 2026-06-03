#!/usr/bin/env bash
# Aggregate benchmark results (called from Snakemake collect rule).
set -euo pipefail

WORKFLOW_ROOT="${1:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}"
CSV_PATH="${2:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}"
MD_PATH="${3:?usage: collect_shell.sh WORKFLOW_ROOT CSV_PATH MD_PATH}"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

PY="${MMML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

exec "$PY" "$WORKFLOW_ROOT/scripts/collect_benchmark.py" \
  --results-root "$WORKFLOW_ROOT/results" \
  --csv "$CSV_PATH" \
  --md "$MD_PATH" \
  --config "$WORKFLOW_ROOT/config.yaml"
