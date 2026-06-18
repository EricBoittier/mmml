#!/usr/bin/env bash
# Launch mmml pycharmm / MLpot under the OpenMPI build linked to libcharmm.so.
#
# MPI ranks: default 1 (recommended). Override for experiments:
#   MMML_MPI_NP=4 ./scripts/mmml-charmm-mpirun.sh md-system ...
# MLpot disables CHARMM domdec and runs the Python callback on every rank — np>1
# is experimental. Rank-0 MLpot bridge (mpi_bridge) runs PhysNet on rank 0 only
# and broadcasts forces; disable with MMML_MLPOT_RANK0_BRIDGE=0.
#
# Crash diagnostics (enabled by default):
#   OMPI_MCA_orte_abort_print_stack=1  — backtrace on MPI abort
#   MMML_MPI_GDB=1                     — gdb batch backtrace (np must be 1)
#   MMML_MPI_VERBOSE=1                 — verbose PRRTE/PLM launch
#   MMML_NO_MPI_ABORT_STACK=1          — disable the above MCA flags
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

export OMPI_MCA_opal_cuda_support=0
export MMML_NO_JAX_COMPILE_THREADS="${MMML_NO_JAX_COMPILE_THREADS:-1}"
PY="${MMML_PYTHON}"
while IFS= read -r line; do
  [[ -n "$line" ]] && eval "$line"
done < <("$PY" - <<'PY'
import os
import shlex

from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_mpirun_path, mpi_shell_setup_lines
from mmml.interfaces.pycharmmInterface.jax_compile_threads import sanitize_xla_flags_env
from mmml.utils.jax_gpu_warmup import ensure_jax_cuda_runtime_libs

sanitize_xla_flags_env(quiet=True)

for line in mpi_shell_setup_lines():
    print(line)

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

MPIRUN_EXTRA=()
while IFS= read -r arg; do
  [[ -n "$arg" ]] && MPIRUN_EXTRA+=("$arg")
done < <("$PY" - <<'PY'
from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_mpirun_extra_args
for arg in mpi_mpirun_extra_args():
    print(arg)
PY
)

MPIRUN="${MMML_MPIRUN:-mpirun}"
MPI_NP="${MMML_MPI_NP:-1}"
if ! [[ "$MPI_NP" =~ ^[1-9][0-9]*$ ]]; then
  echo "mmml-charmm-mpirun: MMML_MPI_NP must be a positive integer (got: ${MMML_MPI_NP:-<unset>})" >&2
  exit 1
fi
if [[ "$MPI_NP" -gt 1 ]]; then
  echo "mmml-charmm-mpirun: warning: MMML_MPI_NP=${MPI_NP} (experimental; domdec off; rank-0 MLpot bridge)" >&2
fi

mmml_mpi_finish() {
  local rc=$1
  local label=$2
  "$PY" - <<PY || true
from mmml.interfaces.pycharmmInterface.charmm_mpi import explain_mpi_crash
explain_mpi_crash(${rc}, argv0=${label@Q})
PY
  exit "$rc"
}

mmml_mpi_run() {
  local -a cmd=( "$MPIRUN" -np "$MPI_NP" "${MPIRUN_EXTRA[@]}" "$@" )
  echo "mmml-charmm-mpirun: ${cmd[*]}" >&2
  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  return "$rc"
}

mmml_gdb_run() {
  local label=$1
  shift
  if [[ "$MPI_NP" != 1 ]]; then
    echo "mmml-charmm-mpirun: MMML_MPI_GDB requires MMML_MPI_NP=1" >&2
    exit 2
  fi
  if ! command -v gdb >/dev/null 2>&1; then
    echo "mmml-charmm-mpirun: gdb not found in PATH" >&2
    exit 2
  fi
  echo "mmml-charmm-mpirun: gdb batch backtrace for ${label}" >&2
  set +e
  gdb -batch \
    -ex "set pagination off" \
    -ex run \
    -ex "thread apply all bt" \
    -ex quit \
    --args "$@"
  local rc=$?
  set -e
  mmml_mpi_finish "$rc" "$label"
}

if [[ "${1:-}" == *.py ]]; then
  if [[ "${MMML_MPI_GDB:-}" == 1 ]]; then
    mmml_gdb_run "python $*" "$PY" "$@"
  fi
  mmml_mpi_run "$PY" "$@"
  mmml_mpi_finish "$?" "python $*"
fi

if [[ "${1:-}" == python || "${1:-}" == python3 ]]; then
  shift
  if [[ "${MMML_MPI_GDB:-}" == 1 ]]; then
    mmml_gdb_run "python $*" "$PY" "$@"
  fi
  mmml_mpi_run "$PY" "$@"
  mmml_mpi_finish "$?" "python $*"
fi

if [[ -n "${MMML_BIN:-}" && -x "${MMML_BIN}" ]]; then
  if [[ "${MMML_MPI_GDB:-}" == 1 ]]; then
    echo "mmml-charmm-mpirun: MMML_MPI_GDB not supported with MMML_BIN" >&2
    exit 2
  fi
  mmml_mpi_run "$MMML_BIN" "$@"
  mmml_mpi_finish "$?" "${MMML_BIN##*/} $*"
fi

if [[ "${MMML_MPI_GDB:-}" == 1 ]]; then
  mmml_gdb_run "mmml $*" "$PY" -m mmml.cli.__main__ "$@"
fi

mmml_mpi_run "$PY" -m mmml.cli.__main__ "$@"
mmml_mpi_finish "$?" "mmml $*"
