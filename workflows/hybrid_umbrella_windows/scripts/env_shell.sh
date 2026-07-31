#!/usr/bin/env bash
# Resolve MMML env (+ CUDA on GPU Slurm jobs), then exec remaining argv.
#
#   bash env_shell.sh -- uv run mmml ...
#   bash env_shell.sh --no-cuda -- bash scripts/run_assemble.sh ...
set -euo pipefail

WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
cd "$REPO_ROOT"

USE_CUDA=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cuda) USE_CUDA=0; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export PYTHONUNBUFFERED=1

if [[ "$USE_CUDA" == "1" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    _part="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
    if [[ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" || "${_part}" == *gpu* ]]; then
      case "${JAX_PLATFORMS:-}" in
        ""|rocm|ROCM) export JAX_PLATFORMS=cuda ;;
      esac
      export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-gpu}"
      export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-gpu}"
      export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-gpu}"
    fi
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" || "${MMML_EXAMPLE_DEVICE:-}" == "gpu" ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-gpu}"
  fi
fi

_cfg_raw="${MMML_WORKFLOW_CONFIG:-$WORKFLOW_ROOT/config.yaml}"
if [[ "$_cfg_raw" = /* ]]; then
  export MMML_WORKFLOW_CONFIG="$_cfg_raw"
else
  export MMML_WORKFLOW_CONFIG="$WORKFLOW_ROOT/$_cfg_raw"
fi

# Prefer workflow checkpoint over a stale interactive MMML_CKPT.
if [[ -f "$MMML_WORKFLOW_CONFIG" ]]; then
  _ckpt="$("$MMML_PYTHON" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${WORKFLOW_ROOT}/scripts')
from campaign_lib import load_config, checkpoint_path
print(checkpoint_path(load_config('${MMML_WORKFLOW_CONFIG}')))
" 2>/dev/null || true)"
  if [[ -n "${_ckpt:-}" ]]; then
    if [[ "$_ckpt" != /* ]]; then
      _ckpt="$REPO_ROOT/$_ckpt"
    fi
    export MMML_CKPT="$_ckpt"
  fi
fi

# CHARMM lib for make-box / hybrid PSF load.
if [[ -z "${CHARMM_LIB_DIR:-}" || ! -d "${CHARMM_LIB_DIR:-}" ]]; then
  eval "$(
    "$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 500 --pbc --box-size 32 \
      2>/dev/null | grep '^export CHARMM_LIB_DIR=' || true
  )" || true
fi
if [[ -n "${CHARMM_LIB_DIR:-}" ]]; then
  if [[ ! -e "${CHARMM_LIB_DIR}/libcharmm.so" && -e "${CHARMM_LIB_DIR}/lib/libcharmm.so" ]]; then
    export CHARMM_LIB_DIR="${CHARMM_LIB_DIR}/lib"
  fi
fi
if [[ -z "${CHARMM_LIB_DIR:-}" ]]; then
  for cand in \
    "$REPO_ROOT/setup/charmm/lib" \
    "$HOME/.cache/mmml-charmm-build/tier_56000000_nodomdec/lib"
  do
    if [[ -e "$cand/libcharmm.so" ]]; then
      export CHARMM_LIB_DIR="$cand"
      break
    elif [[ -e "$cand/lib/libcharmm.so" ]]; then
      export CHARMM_LIB_DIR="$cand/lib"
      break
    fi
  done
fi

exec "$@"
