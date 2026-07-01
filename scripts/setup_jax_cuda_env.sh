#!/usr/bin/env bash
# Prepend pip-shipped cuDNN/cuBLAS to LD_LIBRARY_PATH before mmml imports JAX.
#
# JAX CUDA prefers pip-shipped cuDNN; module-loaded cuDNN can win on LD_LIBRARY_PATH.
# After sourcing, prefer: module unload cudnn  (or load a cuDNN version compatible with JAX).
#
# Usage:
#   source scripts/setup_jax_cuda_env.sh
#   ./scripts/run_dcm9_stability.sh
#
# Install wheels if missing:
#   uv sync --extra gpu
#   # or: pip install 'nvidia-cudnn-cu13>=9.10.1'

_setup_jax_cuda_env() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  local py="${MMML_PYTHON:-$repo_root/.venv/bin/python}"
  if [[ ! -x "$py" ]]; then
    py="$(command -v python3 || true)"
  fi
  if [[ -z "$py" ]]; then
    echo "setup_jax_cuda_env: no python found (set MMML_PYTHON)" >&2
    return 1
  fi

  eval "$("$py" - <<'PY'
import os
import shlex
from mmml.utils.jax_gpu_warmup import ensure_jax_cuda_runtime_libs, jax_cuda_runtime_libs_warning

ensure_jax_cuda_runtime_libs(quiet=True)
ld = os.environ.get("LD_LIBRARY_PATH", "")
if ld:
    print(f"export LD_LIBRARY_PATH={shlex.quote(ld)}")
warning = jax_cuda_runtime_libs_warning(prefix="setup_jax_cuda_env")
if warning:
    print(warning, file=__import__('sys').stderr)
PY
)"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  _setup_jax_cuda_env
else
  _setup_jax_cuda_env
fi
