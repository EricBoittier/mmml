#!/usr/bin/env bash
# Launch on pc-bach (or any CPU Slurm cluster) using profiles/slurm-cpu.
# Usage: snakemake_slurm_cpu.sh [MAX_JOBS]
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
export KARML_SNAKEMAKE_PROFILE="${KARML_SNAKEMAKE_PROFILE:-profiles/slurm-cpu}"
export KARML_WORKFLOW_CONFIG="${KARML_WORKFLOW_CONFIG:-config.pc-bach.cpu.yaml}"
export KARML_CLUSTER="${KARML_CLUSTER:-pc-bach}"
# shellcheck source=../../../scripts/pc_bach_env.sh
source "$REPO_ROOT/scripts/pc_bach_env.sh"
exec bash "$WORKFLOW_ROOT/scripts/snakemake_slurm.sh" "$@"
