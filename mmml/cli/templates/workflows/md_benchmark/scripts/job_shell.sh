#!/usr/bin/env bash
# Run one benchmark job (mmml configure template).
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../../.." && pwd)"
JOB_ID="${1:?usage: job_shell.sh JOB_ID}"
cd "$REPO_ROOT"
PY="${MMML_PYTHON:-python3}"
MMML="${MMML_BIN:-mmml}"
export MMML_CKPT="${MMML_CKPT:?export MMML_CKPT}"
mkdir -p "$WORKFLOW_ROOT/results/$JOB_ID"
exec "$MMML" md-system \
  --config "$WORKFLOW_ROOT/config.yaml" \
  --job-id "$JOB_ID" \
  --output-dir "$WORKFLOW_ROOT/results/$JOB_ID" \
  --quiet
