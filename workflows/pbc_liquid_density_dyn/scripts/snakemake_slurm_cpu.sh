#!/usr/bin/env bash
# Launch on pc-bach (or any CPU Slurm cluster) using profiles/slurm-cpu.
# Usage: snakemake_slurm_cpu.sh [MAX_JOBS]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
export MMML_SNAKEMAKE_PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm-cpu}"
export MMML_WORKFLOW_CONFIG="${MMML_WORKFLOW_CONFIG:-config.pc-bach.cpu.yaml}"
export MMML_CLUSTER="${MMML_CLUSTER:-pc-bach}"
# shellcheck source=../../../scripts/pc_bach_env.sh
source "$REPO_ROOT/scripts/pc_bach_env.sh"
exec bash "$WORKFLOW_ROOT/scripts/snakemake_slurm.sh" "$@"
