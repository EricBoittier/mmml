#!/usr/bin/env bash
# Launch pbc_solvent_burst on Slurm via Snakemake executor plugin.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

JOBS="${1:-10}"
shift || true

exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources gpu=10 charmm_slot=10 \
  --keep-going \
  "$@"
