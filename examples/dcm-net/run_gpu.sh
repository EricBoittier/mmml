#!/usr/bin/env bash
# Robust GPU launcher for JAX: tries module + env combos until one works.
# Usage:
#   ./run_gpu.sh [--train-args "<args to pass to train.py>"] [--max-combos N] [--sanity-only]
# Notes:
# - Avoids nounset around conda activation hooks.
# - Logs attempts into ./_gpu_try_logs/

set -Eeuo pipefail

################################################################################
# CONFIG / ARGS
################################################################################
TRAIN_ARGS=""
MAX_COMBOS=50
SANITY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-args) TRAIN_ARGS="$2"; shift 2;;
    --max-combos) MAX_COMBOS="$2"; shift 2;;
    --sanity-only) SANITY_ONLY=1; shift 1;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

LOG_DIR="_gpu_try_logs"
mkdir -p "$LOG_DIR"

timestamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
log() { echo "[$(timestamp)] $*"; }
hr() { printf '%*s\n' 80 '' | tr ' ' '-'; }

have() { command -v "$1" >/dev/null 2>&1; }

################################################################################
# MODULE HELPERS (Lmod / Environment Modules)
################################################################################
MODULE_SETS=()

module_cmd() {
  if have module; then
    module "$@"
  elif [[ -f /usr/share/Modules/init/sh ]]; then
    # shellcheck disable=SC1091
    . /usr/share/Modules/init/sh
    module "$@"
  else
    return 1
  fi
}

module_available_versions() {
  local prefix="$1"
  module_cmd avail 2>&1 | sed -n "s/.*\( ${prefix}[A-Za-z0-9.\-]*\).*/\1/p" | tr -d ' ' | sort -V -r | uniq
}

try_load_module() {
  local name="$1"
  module_cmd is-loaded "$name" >/dev/null 2>&1 && return 0
  module_cmd load "$name" >/dev/null 2>&1
}

safe_module_purge() {
  module_cmd purge >/dev/null 2>&1 || true
}

################################################################################
# ENV HELPERS (conda, venv)
################################################################################
conda_activate() {
  set +u
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || return 1
  conda activate "$1" >/dev/null 2>&1
  local rc=$?
  set -u
  return $rc
}

venv_activate() {
  # shellcheck disable=SC1090
  source "$1/bin/activate" >/dev/null 2>&1
}

list_conda_envs() {
  set +u
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || { set -u; return 0; }
  set -u
  conda info --envs 2>/dev/null | awk 'NF>0 && $1 !~ /^#/{print $1}' | sed 's/*//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

active_conda_env_name() {
  python - <<'PY' 2>/dev/null || true
import os
print(os.environ.get("CONDA_DEFAULT_ENV",""))
PY
}

################################################################################
# PYTHON + JAX SANITY CHECK
################################################################################
sanity_py() {
  cat <<'PY'
import os, json, sys
res = {"ok": False}
try:
    import jax
    res["python_exe"] = sys.executable
    res["python_ver"] = sys.version.split()[0]
    res["jax"] = getattr(jax, "__version__", "unknown")
    try:
        import jaxlib
        from jaxlib import version as jlv
        res["jaxlib"] = getattr(jaxlib, "__version__", "unknown")
        res["cuda_version_from_jaxlib"] = getattr(jlv, "__cuda_version__", None)
    except Exception as e:
        res["jaxlib_error"] = repr(e)

    res["backend"] = jax.default_backend()
    res["devices"] = [str(d) for d in jax.devices()]
    res["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    ok = (res["backend"] in ("gpu", "cuda")) or any("GPU" in d.upper() or "CUDA" in d.upper() for d in res["devices"])
    res["ok"] = bool(ok)
except Exception as e:
    res["error"] = repr(e)

print(json.dumps(res))
PY
}

run_sanity() {
  local which_python="$1"
  local tag="$2"
  local out="$LOG_DIR/sanity_${tag//[^A-Za-z0-9._-]/_}.json"

  "$which_python" - <<PY >"$out" 2>&1
$(sanity_py)
PY

  if grep -q '"ok": true' "$out"; then
    log "SANITY PASS for [$tag]"
    cat "$out"
    return 0
  else
    log "SANITY FAIL for [$tag] (details in $out)"
    tail -n +1 "$out" | sed 's/^/  /'
    return 1
  fi
}

################################################################################
# TRAIN RUNNER
################################################################################
run_training() {
  local which_python="$1"
  if [[ $SANITY_ONLY -eq 1 ]]; then
    log "--sanity-only set; skipping training."
    return 0
  fi

  if have uv && [[ -f "pyproject.toml" ]]; then
    log "Launching training via: uv run python train.py $TRAIN_ARGS"
    uv run python train.py $TRAIN_ARGS
  else
    log "Launching training via: $which_python train.py $TRAIN_ARGS"
    "$which_python" train.py $TRAIN_ARGS
  fi
}

################################################################################
# PREP: baseline env vars and optional site-packages isolation
################################################################################
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module_cmd avail >/dev/null 2>&1 && {
  if module_available_versions "Miniconda" | head -n1 >/dev/null; then
    try_load_module Miniconda3 || true
  fi
}

################################################################################
# DISCOVER MODULES, ENVS, CANDIDATES
################################################################################
CUDA_LIST=()
CUDNN_LIST=()

if module_cmd avail >/dev/null 2>&1; then
  mapfile -t CUDA_LIST < <(module_available_versions "CUDA/")
  mapfile -t CUDNN_LIST < <(module_available_versions "cuDNN/")
fi

CUDA_PREF=()
for c in "${CUDA_LIST[@]}"; do
  if [[ "$c" =~ ^CUDA/12 ]]; then CUDA_PREF+=("$c"); fi
done
for c in "${CUDA_LIST[@]}"; do
  if [[ ! "$c" =~ ^CUDA/12 ]]; then CUDA_PREF+=("$c"); fi
done

choose_cudnn_for_cuda() {
  local cuda="$1"
  local ver="${cuda#CUDA/}"
  for d in "${CUDNN_LIST[@]}"; do
    if [[ "$d" == *"CUDA-${ver}"* ]]; then
      echo "$d"
      return 0
    fi
  done
  [[ ${#CUDNN_LIST[@]} -gt 0 ]] && echo "${CUDNN_LIST[0]}"
}

MODULE_SETS+=("none")
for cu in "${CUDA_PREF[@]}"; do
  cudnn="$(choose_cudnn_for_cuda "$cu")"
  if [[ -n "$cudnn" ]]; then
    MODULE_SETS+=("$cu $cudnn")
  else
    MODULE_SETS+=("$cu")
  fi
done

ENV_CANDIDATES=()
ACTIVE_ENV="$(active_conda_env_name || true)"
[[ -n "$ACTIVE_ENV" ]] && ENV_CANDIDATES+=("conda:$ACTIVE_ENV")
ENV_CANDIDATES+=("conda:mmml-full")

if have conda; then
  while IFS= read -r e; do
    [[ -z "$e" ]] && continue
    [[ "$e" == "$ACTIVE_ENV" ]] && continue
    [[ "$e" == "mmml-full" ]] && continue
    ENV_CANDIDATES+=("conda:$e")
  done < <(list_conda_envs || true)
fi

[[ -d "$HOME/mmml/.venv" ]] && ENV_CANDIDATES+=("venv:$HOME/mmml/.venv")
ENV_CANDIDATES+=("system:")

################################################################################
# TRY COMBINATIONS
################################################################################
COMBOS_TRIED=0

for mset in "${MODULE_SETS[@]}"; do
  if module_cmd avail >/dev/null 2>&1; then
    safe_module_purge || true
    if [[ "$mset" != "none" ]]; then
      log "Trying modules: $mset"
      for m in $mset; do
        try_load_module "$m" || log "  (could not load $m; continuing anyway)"
      done
    else
      log "Trying with NO CUDA/cuDNN modules"
    fi
  else
    [[ "$mset" == "none" ]] || continue
  fi

  for env in "${ENV_CANDIDATES[@]}"; do
    (( COMBOS_TRIED+=1 ))
    if (( COMBOS_TRIED > MAX_COMBOS )); then
      log "Reached --max-combos=$MAX_COMBOS; stopping search."
      break 2
    fi

    set +u
    if have conda; then eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true; fi
    set -u

    if [[ -n "${VIRTUAL_ENV-}" ]]; then deactivate || true; fi

    TAG=""
    WHICH_PY=""

    case "$env" in
      conda:*)
        name="${env#conda:}"
        TAG="modules=[$mset] env=conda:$name"
        log "Activating conda env: $name"
        if conda_activate "$name"; then
          WHICH_PY="$(command -v python || true)"
        else
          log "  (could not activate conda env $name)"
          continue
        fi
        ;;
      venv:*)
        path="${env#venv:}"
        TAG="modules=[$mset] env=venv:$path"
        log "Activating venv: $path"
        if [[ -f "$path/bin/activate" ]]; then
          venv_activate "$path" || { log "  (venv activate failed)"; continue; }
          WHICH_PY="$(command -v python || true)"
        else
          log "  (venv not found at $path)"
          continue
        fi
        ;;
      system:*)
        TAG="modules=[$mset] env=system"
        WHICH_PY="$(command -v python || command -v python3 || true)"
        ;;
      *)
        continue
        ;;
    esac

    if [[ -z "$WHICH_PY" ]]; then
      log "No python found for $TAG"
      continue
    fi

    log "Using python: $WHICH_PY"
    if have nvidia-smi; then
      log "nvidia-smi:"
      nvidia-smi || true
    fi

    if run_sanity "$WHICH_PY" "$TAG"; then
      hr
      log "🎉 Found a working GPU JAX setup: $TAG"
      hr
      run_training "$WHICH_PY"
      exit $?
    fi
  done
done

hr
log "❌ No working GPU-enabled JAX environment found."
log "See logs under: $LOG_DIR/"
echo "Tips:"
echo "  • Ensure your jaxlib build matches your CUDA major version."
echo "  • Try loading a different CUDA module (e.g., 12.4 vs 12.2) that matches jaxlib’s reported __cuda_version__."
echo "  • If you rely on uv, keep a pyproject.toml with your deps so uv can resolve them."
echo "  • Consider reinstalling jax/jaxlib inside the target env with the correct CUDA build."
exit 1

