#!/usr/bin/env bash
# Prepare one DCM:3 trimer geometry (Snakemake helper).
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
GEOM_ID="${1:?usage: prepare_geometry_shell.sh GEOM_ID}"

cd "$REPO_ROOT"
PY="${MMML_PYTHON:-}"
if [[ -z "$PY" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
exec "$MPIRUN" "$PY" "$WORKFLOW_ROOT/scripts/prepare_geometry.py" "$GEOM_ID" \
  --config "$WORKFLOW_ROOT/config.yaml"
