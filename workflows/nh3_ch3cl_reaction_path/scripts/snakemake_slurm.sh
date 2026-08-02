#!/usr/bin/env bash
# Launch nh3_ch3cl_reaction_path on the studix GPU queue.
# Usage: bash scripts/snakemake_slurm.sh [MAX_JOBS] [extra snakemake args...]
#
#   nohup bash scripts/snakemake_slurm.sh 8 > snakemake_gpu.log 2>&1 &
#   MMML_WORKFLOW_CONFIG=config.smoke.yaml bash scripts/snakemake_slurm.sh 4
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
PY="${MMML_PYTHON}"

PROFILE="${MMML_SNAKEMAKE_PROFILE:-profiles/slurm}"
_cfg_raw="${MMML_WORKFLOW_CONFIG:-config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG_PATH="$_cfg_raw"
elif [[ "$_cfg_raw" == */* ]]; then
  CFG_PATH="$(cd "$(dirname "$_cfg_raw")" && pwd)/$(basename "$_cfg_raw")"
else
  CFG_PATH="${WORKFLOW_ROOT}/${_cfg_raw}"
fi
export MMML_WORKFLOW_CONFIG="$CFG_PATH"
# Snakemake profile lists these as required envvars; empty is fine for dry-run.
export CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-}"
export MMML_CKPT="${MMML_CKPT:-}"
export MMML_CGENFF_EXTRA_RTF="${MMML_CGENFF_EXTRA_RTF:-}"
export MMML_CGENFF_EXTRA_PRM="${MMML_CGENFF_EXTRA_PRM:-}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-}"
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

if [[ -z "${DEFAULT_RES// }" ]]; then
  echo "ERROR: could not resolve Slurm resources from ${CFG_PATH}" >&2
  exit 1
fi

JOBS="${1:-$DEFAULT_JOBS}"
shift || true

UV="${MMML_UV:-uv}"
echo "Snakemake Slurm: profile=${PROFILE} config=${CFG_PATH} -j${JOBS} --resources ${DEFAULT_RES}" >&2
echo "  checkpoint=$(${PY} -c "import sys; sys.path.insert(0,'${WORKFLOW_ROOT}/scripts'); from campaign_lib import load_config, checkpoint_path; print(checkpoint_path(load_config('${CFG_PATH}')))")" >&2

# shellcheck disable=SC2086
exec "$UV" run --with snakemake --with snakemake-executor-plugin-slurm snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
