#!/usr/bin/env bash
# Launch mmml pycharmm / MLpot under the OpenMPI build linked to libcharmm.so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

PY="${MMML_PYTHON}"
while IFS= read -r line; do
  [[ -n "$line" ]] && eval "$line"
done < <("$PY" - <<'PY'
import os
import shlex

from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_mpirun_path, mpi_shell_setup_lines
from mmml.utils.jax_gpu_warmup import ensure_jax_cuda_runtime_libs

for line in mpi_shell_setup_lines():
    print(line)

# After MPI/CHARMM prepends LD_LIBRARY_PATH, prefer pip cuDNN >= 9.10.1 over module stacks.
bundled = ensure_jax_cuda_runtime_libs(quiet=True)
ld = os.environ.get("LD_LIBRARY_PATH", "")
if ld:
    print(f"export LD_LIBRARY_PATH={shlex.quote(ld)}")
if not bundled:
    print(
        "echo mmml-charmm-mpirun: warning: no pip nvidia/cudnn libs; "
        "JAX CUDA may fail if module cuDNN is < 9.10.1. "
        "Run: uv sync --extra gpu >&2",
    )

mpirun = charmm_mpirun_path()
if mpirun is None:
    raise SystemExit(
        "mmml-charmm-mpirun: could not find OpenMPI mpirun for libcharmm.so. "
        "Set MMML_MPIRUN=/path/to/mpirun or MMML_MPI_LD_PATH_EXTRA."
    )
PY
)

MPIRUN="${MMML_MPIRUN:-mpirun}"
if [[ "${1:-}" == *.py ]]; then
  exec "$MPIRUN" -np 1 "$PY" "$@"
fi
if [[ "${1:-}" == python || "${1:-}" == python3 ]]; then
  shift
  exec "$MPIRUN" -np 1 "$PY" "$@"
fi
if [[ -n "${MMML_BIN:-}" && -x "${MMML_BIN}" ]]; then
  exec "$MPIRUN" -np 1 "$MMML_BIN" "$@"
fi
exec "$MPIRUN" -np 1 "$PY" -m mmml.cli.__main__ "$@"
