#!/usr/bin/env bash
# Launch dcm_heat_scaling on Slurm via Snakemake executor plugin.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

JOBS="${1:-4}"
shift || true

if ! uv run --with snakemake --with snakemake-executor-plugin-slurm \
  python -c "import snakemake_executor_plugin_slurm" 2>/dev/null; then
  echo "snakemake-executor-plugin-slurm is not available." >&2
  echo "Install with:" >&2
  echo "  uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake --version" >&2
  exit 1
fi

exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources gpu=1 mpi=1 \
  --keep-going \
  "$@"
