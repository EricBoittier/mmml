#!/usr/bin/env bash
# Launch dcm_density_setup_compare on Slurm via Snakemake executor plugin.
#
# Usage: snakemake_slurm.sh [MAX_JOBS] [extra snakemake args...]
#   MAX_JOBS defaults to tier pools from the workflow config (fast + slow when tiering on).
#   Snakemake flags may be passed without a leading job count, e.g.:
#     bash scripts/snakemake_slurm.sh --forcerun run_setup_compare
#
# Prep sweep (must export config to compute jobs via MMML_WORKFLOW_CONFIG):
#   MMML_WORKFLOW_CONFIG=config.prep_sweep.yaml bash scripts/snakemake_slurm.sh
#   bash scripts/snakemake_prep_sweep.sh
#
# From pc-studix login node (no OpenCL on login — jobs run on GPU compute nodes):
#   export MMML_CKPT=/path/to/DESdimers_params.json
#   nohup bash scripts/snakemake_slurm.sh > snakemake_slurm.log 2>&1 &
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

# Forwarded to Slurm jobs via profiles/slurm/config.yaml envvars — must exist on driver start.
export MMML_CKPT="${MMML_CKPT:-${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json}"
if [[ ! -f "${MMML_CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${MMML_CKPT}" >&2
  echo "  export MMML_CKPT=${REPO_ROOT}/examples/ckpts_json/DESdimers_params.json" >&2
  exit 1
fi
export MMML_CKPT="$(readlink -f "${MMML_CKPT}")"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
echo "snakemake_slurm.sh: MMML_CKPT=${MMML_CKPT}" >&2

_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
elif [[ "$_cfg_raw" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$_cfg_raw")" && pwd)/$(basename "$_cfg_raw")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
CONFIG_ARGS=(--configfile "$CFG_PATH")

IFS=$'\t' read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, slurm_launch_jobs, slurm_resources_cli
cfg = load_config(Path('${CFG_PATH}'))
print(f\"{slurm_launch_jobs(cfg)}\t{slurm_resources_cli(cfg)}\")
")
EOF

if [[ -z "${DEFAULT_RES:-}" ]]; then
  echo "snakemake_slurm.sh: failed to resolve --resources from ${CFG_PATH}" >&2
  exit 1
fi

JOBS="$DEFAULT_JOBS"
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  JOBS="$1"
  shift || true
fi

if [[ "${MMML_SNAKEMAKE_FORCE:-}" != "1" ]]; then
  _existing=()
  while IFS= read -r _pid; do
    _cwd="$(readlink -f "/proc/${_pid}/cwd" 2>/dev/null || true)"
    if [[ "$_cwd" == "$WORKFLOW_ROOT" ]]; then
      _existing+=("$_pid")
    fi
  done < <(pgrep -f 'snakemake --profile profiles/slurm|uv run --with snakemake' 2>/dev/null || true)
  if ((${#_existing[@]} > 0)); then
    echo "snakemake_slurm.sh: driver already running in ${WORKFLOW_ROOT} (PIDs: ${_existing[*]})." >&2
    echo "  bash scripts/stop_snakemake.sh" >&2
    echo "  uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake --profile profiles/slurm --unlock" >&2
    echo "  Or force a second driver: MMML_SNAKEMAKE_FORCE=1 bash scripts/snakemake_slurm.sh ..." >&2
    exit 1
  fi
fi

echo "Snakemake Slurm: config=${CFG_PATH} -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec uv run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile profiles/slurm \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
