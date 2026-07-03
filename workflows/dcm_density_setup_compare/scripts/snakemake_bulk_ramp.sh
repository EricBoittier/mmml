#!/usr/bin/env bash
# Launch bulk-density ramp (config.bulk_ramp.yaml) on Slurm.
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MMML_WORKFLOW_CONFIG="${MMML_WORKFLOW_CONFIG:-config.bulk_ramp.yaml}"
exec bash "$WORKFLOW_ROOT/scripts/snakemake_slurm.sh" "$@"
