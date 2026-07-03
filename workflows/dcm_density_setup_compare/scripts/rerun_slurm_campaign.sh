#!/usr/bin/env bash
# Stop the Snakemake driver and relaunch all workflow cells with the current MMML_CKPT.
#
# Usage (pc-studix / gpu09 login node):
#   export MMML_CKPT=/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1/dcm1_params.json
#   bash scripts/rerun_slurm_campaign.sh              # prep sweep (24 jobs)
#   bash scripts/rerun_slurm_campaign.sh --main         # main config.yaml matrix
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
# shellcheck source=ckpt_defaults.sh
source "$WORKFLOW_ROOT/scripts/ckpt_defaults.sh"

MODE="prep"
LOG="snakemake_prep_sweep.log"
LAUNCHER="snakemake_prep_sweep.sh"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --main)
      MODE="main"
      LOG="snakemake_slurm.log"
      LAUNCHER="snakemake_slurm.sh"
      shift
      ;;
    --prep-sweep)
      MODE="prep"
      LOG="snakemake_prep_sweep.log"
      LAUNCHER="snakemake_prep_sweep.sh"
      shift
      ;;
    *)
      echo "Unknown option: $1 (use --prep-sweep or --main)" >&2
      exit 1
      ;;
  esac
done

export MMML_CKPT="${MMML_CKPT:-$(default_mmml_ckpt "$REPO_ROOT")}"
export MMML_CKPT="$(readlink -f "${MMML_CKPT}")"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

if [[ ! -f "${MMML_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${MMML_CKPT}" >&2
  echo "  export MMML_CKPT=/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1/dcm1_params.json" >&2
  exit 1
fi

if [[ "$MODE" == prep ]]; then
  export MMML_WORKFLOW_CONFIG="${WORKFLOW_ROOT}/config.prep_sweep.yaml"
  MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml bash scripts/preflight.sh
  CONFIGFILE=(--configfile config.prep_sweep.yaml)
else
  unset MMML_WORKFLOW_CONFIG || true
  export MMML_WORKFLOW_CONFIG="${WORKFLOW_ROOT}/config.yaml"
  bash scripts/preflight.sh
  CONFIGFILE=(--configfile config.yaml)
fi

echo "rerun_slurm_campaign.sh: mode=${MODE} MMML_CKPT=${MMML_CKPT}" >&2

bash scripts/stop_snakemake.sh 2>/dev/null || true

uv run --with snakemake --with snakemake-executor-plugin-slurm \
  snakemake --profile profiles/slurm \
  "${CONFIGFILE[@]}" \
  --unlock

nohup bash "scripts/${LAUNCHER}" --forcerun run_setup_compare >"${LOG}" 2>&1 &
echo "Launched driver PID $! — monitor: tail -f ${WORKFLOW_ROOT}/${LOG}" >&2
echo "  pgrep -af 'snakemake --profile profiles/slurm'" >&2
