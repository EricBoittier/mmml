#!/usr/bin/env bash
# Launch pbc_liquid_density_dyn locally on this node's GPUs (no Slurm).
#
# Usage:
#   bash scripts/snakemake_local.sh [MAX_JOBS] [snakemake args...]
#
# Examples:
#   KARML_WORKFLOW_CONFIG=config.yaml bash scripts/snakemake_local.sh
#   KARML_WORKFLOW_CONFIG=config.gpu08.local.yaml nohup bash scripts/snakemake_local.sh >> snakemake_local.log 2>&1 &
#
# Uses profiles/local (executor: local). Each job pins to one GPU via with_local_gpu.sh.
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

PROFILE="${KARML_SNAKEMAKE_PROFILE:-profiles/local}"
_cfg_raw="${KARML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
elif [[ "$_cfg_raw" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$_cfg_raw")" && pwd)/$(basename "$_cfg_raw")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export KARML_WORKFLOW_CONFIG="$CFG_PATH"
CONFIG_ARGS=(--configfile "$CFG_PATH")

_LOCK_DIR="${KARML_SNAKEMAKE_LOCK_DIR:-/tmp/karml_snakemake_locks_${USER:-$(id -un)}}"
mkdir -p "$_LOCK_DIR"
_CFG_LOCK="${_LOCK_DIR}/$(basename "$CFG_PATH").driver.lock"
if ! flock -n 9; then
  echo "Snakemake driver already running for ${CFG_PATH} (lock ${_CFG_LOCK})" >&2
  exit 0
fi 9>"$_CFG_LOCK"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export KARML_LOCAL_GPU_PIN="${KARML_LOCAL_GPU_PIN:-1}"
if [[ -z "${KARML_LOCAL_GPU_SLOTS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    KARML_LOCAL_GPU_SLOTS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  KARML_LOCAL_GPU_SLOTS="${KARML_LOCAL_GPU_SLOTS:-2}"
  export KARML_LOCAL_GPU_SLOTS
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

UV="${KARML_UV:-}"
if [[ -z "$UV" || ! -x "$UV" ]]; then
  echo "ERROR: uv not found (set KARML_UV or install uv in ~/.local/bin)" >&2
  exit 1
fi

echo "Snakemake local: host=$(hostname) profile=${PROFILE} config=${CFG_PATH}" >&2
echo "  KARML_CKPT=${KARML_CKPT:-<unset>} GPUs=${KARML_LOCAL_GPU_SLOTS} -j${JOBS} --resources ${DEFAULT_RES}" >&2

# shellcheck disable=SC2086
exec "$UV" run --with snakemake snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
