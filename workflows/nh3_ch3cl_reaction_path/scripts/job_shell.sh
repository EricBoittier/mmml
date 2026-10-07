#!/usr/bin/env bash
# Resolve KARML / CHARMM env, then run run_job.py with the remaining argv.
# Usage (from Snakemake):
#   bash scripts/job_shell.sh --job make_boxes --output-dir ABS --status ABS ...
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_karml_env.sh
source "$REPO_ROOT/scripts/resolve_karml_env.sh"
karml_resolve_env "$REPO_ROOT"
PY="${KARML_PYTHON}"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# Studix GPU partition is NVIDIA. Empty JAX_PLATFORMS (snakemake envvars) or a
# stale rocm value makes ``import jax_md`` fail before umbrella/hybrid can run.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _part="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
  if [[ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" || "${_part}" == *gpu* ]]; then
    case "${JAX_PLATFORMS:-}" in
      ""|rocm|ROCM) export JAX_PLATFORMS=cuda ;;
      *)
        if [[ "${JAX_PLATFORMS}" == *[Rr][Oo][Cc][Mm]* ]]; then
          export JAX_PLATFORMS="$(
            printf '%s' "${JAX_PLATFORMS}" \
              | sed -E 's/(^|,)[Rr][Oo][Cc][Mm](,|$)/\1\2/g; s/,,/,/g; s/^,//; s/,$//'
          )"
          [[ -z "${JAX_PLATFORMS}" ]] && export JAX_PLATFORMS=cuda
        fi
        ;;
    esac
    export KARML_MLPOT_DEVICE="${KARML_MLPOT_DEVICE:-gpu}"
    export KARML_JAX_WARMUP_DEVICE="${KARML_JAX_WARMUP_DEVICE:-gpu}"
  fi
fi

_cfg_raw="${KARML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  CFG="$_cfg_raw"
else
  CFG="$WORKFLOW_ROOT/$_cfg_raw"
fi
export KARML_WORKFLOW_CONFIG="$CFG"

# Ensure a CHARMM build is visible for make-box / ADUMB (PyCHARMM).
if [[ -z "${CHARMM_LIB_DIR:-}" || ! -d "${CHARMM_LIB_DIR:-}" ]]; then
  eval "$(
    "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 500 --pbc --box-size 32 \
      2>/dev/null | grep '^export CHARMM_LIB_DIR=' || true
  )" || true
fi
# Prefer .../lib when a parent directory was exported without libcharmm.so.
if [[ -n "${CHARMM_LIB_DIR:-}" ]]; then
  if [[ ! -e "${CHARMM_LIB_DIR}/libcharmm.so" && -e "${CHARMM_LIB_DIR}/lib/libcharmm.so" ]]; then
    export CHARMM_LIB_DIR="${CHARMM_LIB_DIR}/lib"
  fi
fi
if [[ -z "${CHARMM_LIB_DIR:-}" ]]; then
  for cand in \
    "$REPO_ROOT/setup/charmm/lib" \
    "$REPO_ROOT/setup/charmm" \
    "$HOME/.cache/karml-charmm-build/tier_56000000_nodomdec/lib"
  do
    if [[ -d "$cand" ]] && { [[ -e "$cand/libcharmm.so" ]] || [[ -e "$cand/lib/libcharmm.so" ]]; }; then
      if [[ -e "$cand/lib/libcharmm.so" && ! -e "$cand/libcharmm.so" ]]; then
        export CHARMM_LIB_DIR="$cand/lib"
      else
        export CHARMM_LIB_DIR="$cand"
      fi
      break
    fi
  done
fi

# Always prefer workflow config checkpoint over a stale shell KARML_CKPT
# (e.g. leftover examples/m/kl.json from an interactive session).
_ckpt="$("$PY" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, checkpoint_path
cfg = load_config(Path('${CFG}'))
print(Path('${REPO_ROOT}') / checkpoint_path(cfg))
" 2>/dev/null || true)"
if [[ -n "${_ckpt}" && -e "${_ckpt}" ]]; then
  export KARML_CKPT="$_ckpt"
elif [[ -z "${KARML_CKPT:-}" || ! -e "${KARML_CKPT}" ]]; then
  echo "WARNING: workflow checkpoint unresolved; KARML_CKPT=${KARML_CKPT:-unset}" >&2
fi

echo "=== nh3_ch3cl_reaction_path job_shell ===" >&2
echo "REPO_ROOT=${REPO_ROOT}" >&2
echo "PY=${PY}" >&2
echo "KARML_CKPT=${KARML_CKPT:-<unset>}" >&2
echo "CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-<unset>}" >&2
echo "JAX_PLATFORMS=${JAX_PLATFORMS:-<unset>}" >&2
echo "KARML_WORKFLOW_CONFIG=${KARML_WORKFLOW_CONFIG}" >&2
echo "argv: $*" >&2

exec "$PY" "$WORKFLOW_ROOT/scripts/run_job.py" \
  --config "$CFG" \
  --repo-root "$REPO_ROOT" \
  "$@"
