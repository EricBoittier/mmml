#!/usr/bin/env bash
# Launch pbc_solvent_burst on Slurm via Snakemake executor plugin.
#
# Usage: snakemake_slurm.sh [MAX_CONCURRENT]
#   MAX_CONCURRENT defaults to slurm_max_concurrent in config.yaml (capped at
#   matrix size). Set gpu and charmm_slot pools to the same value for max throughput.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

DEFAULT_JOBS="$("$PY" -c "
import sys
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_max_concurrent
print(slurm_max_concurrent(load_config()))
")"

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

echo "Snakemake Slurm: -j${JOBS} --resources gpu=${JOBS} charmm_slot=${JOBS}" >&2

exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  -j"$JOBS" \
  --resources gpu="$JOBS" charmm_slot="$JOBS" \
  --keep-going \
  "$@"
