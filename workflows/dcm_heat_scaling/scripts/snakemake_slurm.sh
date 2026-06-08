#!/usr/bin/env bash
# Launch dcm_heat_scaling on Slurm via Snakemake executor plugin.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

JOBS="${1:-4}"
shift || true

exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources gpu=1 charmm_slot=1 \
  --keep-going \
  "$@"
