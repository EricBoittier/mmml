#!/usr/bin/env bash
# Launch pbc_liquid_density_dyn locally on this node's GPUs (no Slurm).
#
# Usage:
#   bash scripts/snakemake_local.sh [MAX_JOBS] [snakemake args...]
#
# Examples:
#   MMML_WORKFLOW_CONFIG=config.yaml bash scripts/snakemake_local.sh
#   MMML_WORKFLOW_CONFIG=config.gpu08.local.yaml nohup bash scripts/snakemake_local.sh >> snakemake_local.log 2>&1 &
#
# Uses profiles/local (executor: local). Each job pins to one GPU via with_local_gpu.sh.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/local}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
elif [[ "$_cfg_raw" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$_cfg_raw")" && pwd)/$(basename "$_cfg_raw")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
CONFIG_ARGS=()
if [[ "$CFG_PATH" != "$WORKFLOW_ROOT/config.yaml" ]]; then
  CONFIG_ARGS=(--configfile "$CFG_PATH")
fi

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export MMML_LOCAL_GPU_PIN="${MMML_LOCAL_GPU_PIN:-1}"
if [[ -z "${MMML_CKPT:-}" ]]; then
  _default_ckpt="/mmhome/boittier/home/mmml_tutorial/acodcm/ckpts/dcm1-c137fb42-1f65-4748-880b-8f8184a20f70"
  if [[ -d "$_default_ckpt" ]]; then
    export MMML_CKPT="$_default_ckpt"
  fi
fi

if [[ -z "${MMML_LOCAL_GPU_SLOTS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    MMML_LOCAL_GPU_SLOTS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  MMML_LOCAL_GPU_SLOTS="${MMML_LOCAL_GPU_SLOTS:-2}"
  export MMML_LOCAL_GPU_SLOTS
fi

IFS=$'\t' read -r DEFAULT_JOBS DEFAULT_RES <<EOF
$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, local_launch_jobs, local_resources_cli
cfg = load_config(Path('${CFG_PATH}'))
print(f\"{local_launch_jobs(cfg)}\t{local_resources_cli(cfg)}\")
")
EOF

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

echo "Snakemake local: host=$(hostname) profile=${PROFILE} config=${CFG_PATH}" >&2
echo "  MMML_CKPT=${MMML_CKPT:-<unset>} GPUs=${MMML_LOCAL_GPU_SLOTS} -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec uv run --with snakemake snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
