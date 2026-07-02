#!/usr/bin/env bash
# Launch prep-parameter sweep (config.prep_sweep.yaml) on Slurm.
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MMML_WORKFLOW_CONFIG="${MMML_WORKFLOW_CONFIG:-config.prep_sweep.yaml}"
exec bash "$WORKFLOW_ROOT/scripts/snakemake_slurm.sh" "$@"
