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
# Avoid grabbing the whole GPU up front when several JAX jobs share a node.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

_mmml_cuda_ready() {
  # nvidia-smi must see at least one device in this job's view.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    return 1
  fi
  # Cheap cuInit via jax — matches the failure mode in window logs.
  "$MMML_PYTHON" - <<'PY' >/dev/null 2>&1
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
devs = jax.devices("cuda")
assert devs, "no cuda devices"
print(devs[0])
PY
}

_mmml_wait_cuda() {
  local tries="${MMML_CUDA_INIT_RETRIES:-12}"
  local i=1
  while (( i <= tries )); do
    if _mmml_cuda_ready; then
      echo "[env_shell] CUDA ready (try ${i}/${tries})  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}" >&2
      return 0
    fi
    echo "[env_shell] CUDA not ready (try ${i}/${tries}); sleep ${i}s …  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}" >&2
    sleep "$i"
    i=$((i + 1))
  done
  echo "[env_shell] FAIL: CUDA init failed after ${tries} tries (cuInit / nvidia-smi)." >&2
  echo "[env_shell]   host=$(hostname) job=${SLURM_JOB_ID:-local} gres=${SLURM_JOB_GPUS:-?} cvd=${CUDA_VISIBLE_DEVICES:-}" >&2
  nvidia-smi -L >&2 || true
  return 1
}

if [[ "$USE_CUDA" == "1" ]]; then
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
      export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-gpu}"
      export MMML_MLPOT_DEVICE="${MMML_MLPOT_DEVICE:-gpu}"
      export MMML_JAX_WARMUP_DEVICE="${MMML_JAX_WARMUP_DEVICE:-gpu}"
      _mmml_wait_cuda
    fi
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" || "${MMML_EXAMPLE_DEVICE:-}" == "gpu" ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    export MMML_EXAMPLE_DEVICE="${MMML_EXAMPLE_DEVICE:-gpu}"
    _mmml_wait_cuda
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
