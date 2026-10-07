#!/usr/bin/env bash
# Launch pbc_liquid_density_dyn on Slurm.
# Usage: snakemake_slurm.sh [MAX_JOBS]
#   KARML_SNAKEMAKE_PROFILE=profiles/slurm-cpu KARML_WORKFLOW_CONFIG=config.pc-bach.cpu.yaml ...
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKFLOW_ROOT"

REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

PROFILE="${KARML_SNAKEMAKE_PROFILE:-profiles/slurm}"
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

if [[ -z "${CHARMM_LIB_DIR:-}" ]]; then
  eval "$(
    "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 2660 --pbc --box-size 32 \
      2>/dev/null | grep '^export CHARMM_LIB_DIR=' || true
  )"
fi
export CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-$HOME/.cache/karml-charmm-build/tier_56000000_nodomdec/lib}"
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

UV="${KARML_UV:-}"
if [[ -z "$UV" || ! -x "$UV" ]]; then
  echo "ERROR: uv not found (set KARML_UV or install uv in ~/.local/bin)" >&2
  exit 1
fi

echo "Snakemake Slurm: profile=${PROFILE} config=${CFG_PATH} KARML_CKPT=${KARML_CKPT:-<unset>} -j${JOBS} --resources ${DEFAULT_RES}" >&2

# Stale lock after a killed driver or overlapping launch attempts.
# --no-project: avoid broken namespace packages in .venv (e.g. pyarrow NFS stubs)
# shadowing deps required by snakemake-executor-plugin-slurm/pandas.
# --python 3.12: project .venv is 3.13; uv would otherwise reuse it and still
# import the broken pyarrow namespace from site-packages.
unset VIRTUAL_ENV PYTHONPATH
export PYTHONNOUSERSITE=1
_SNAKE_UV=(run --no-project --python 3.12 --with snakemake --with snakemake-executor-plugin-slurm)
"$UV" "${_SNAKE_UV[@]}" snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  --unlock 2>/dev/null || true

# shellcheck disable=SC2086
exec "$UV" "${_SNAKE_UV[@]}" snakemake \
  --profile "$PROFILE" \
  "${CONFIG_ARGS[@]}" \
  -j"$JOBS" \
  --resources ${DEFAULT_RES} \
  --keep-going \
  "$@"
