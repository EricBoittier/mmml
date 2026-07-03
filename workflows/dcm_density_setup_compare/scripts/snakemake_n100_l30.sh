#!/usr/bin/env bash
# Launch N=100 @ L=30 Å matrix (config.n100_l30.yaml) on Slurm.
set -euo pipefail
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MMML_WORKFLOW_CONFIG="${MMML_WORKFLOW_CONFIG:-config.n100_l30.yaml}"
exec bash "$WORKFLOW_ROOT/scripts/snakemake_slurm.sh" "$@"
