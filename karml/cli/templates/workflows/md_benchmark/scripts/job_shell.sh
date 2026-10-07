#!/usr/bin/env bash
# Run one benchmark job (karml configure template).
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../../.." && pwd)"
JOB_ID="${1:?usage: job_shell.sh JOB_ID}"
cd "$REPO_ROOT"
PY="${KARML_PYTHON:-python3}"
KARML="${KARML_BIN:-karml}"
export KARML_CKPT="${KARML_CKPT:?export KARML_CKPT}"
mkdir -p "$WORKFLOW_ROOT/results/$JOB_ID"
exec "$KARML" md-system \
  --config "$WORKFLOW_ROOT/config.yaml" \
  --job-id "$JOB_ID" \
  --output-dir "$WORKFLOW_ROOT/results/$JOB_ID" \
  --quiet
