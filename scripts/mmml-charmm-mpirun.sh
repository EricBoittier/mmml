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
#
# Orphan cleanup (Ctrl+C / prterun exit can leave rank-0 Python running):
#   ./scripts/mmml-kill-orphans.sh           # dry-run list
#   ./scripts/mmml-kill-orphans.sh --kill    # reap orphans before a new launch
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
preload = os.environ.get("LD_PRELOAD", "")
if preload:
    print(f"export LD_PRELOAD={shlex.quote(preload)}")
if not bundled:
    import sys

    print(
        "mmml-charmm-mpirun: warning: no pip nvidia/cudnn libs; "
        "JAX CUDA may fail if module cuDNN is below 9.10.1. "
        "Run: uv sync --extra gpu",
        file=sys.stderr,
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
  if [[ "${MMML_MLPOT_SPATIAL_MPI:-}" == 1 || "${MMML_MLPOT_SPATIAL_MPI:-}" == true || "${MMML_MLPOT_SPATIAL_MPI:-}" == yes ]]; then
    echo "mmml-charmm-mpirun: MMML_MPI_NP=${MPI_NP} spatial ML (MMML_MLPOT_SPATIAL_MPI=1; 1 GPU per rank)" >&2
  else
    echo "mmml-charmm-mpirun: warning: MMML_MPI_NP=${MPI_NP} (experimental; domdec off; rank-0 MLpot bridge)" >&2
  fi
fi

# Warn when multiple MPI ranks and multi-GPU pmap may oversubscribe GPUs.
_ML_GPU_COUNT="${MMML_MLPOT_N_GPUS:-1}"
for ((i=1; i<${#@}; i++)); do
  if [[ "${!i}" == "--ml-gpu-count" && $((i+1)) -le $# ]]; then
    _ML_GPU_COUNT="${@:$((i+1)):1}"
    break
  fi
done
if [[ "$MPI_NP" -gt 1 && "${_ML_GPU_COUNT:-1}" -gt 1 ]]; then
  echo "mmml-charmm-mpirun: warning: np=${MPI_NP} with ml_gpu_count=${_ML_GPU_COUNT} may fight over GPUs; use --ml-gpu-count 1 and MMML_MLPOT_SPATIAL_MPI=1 or np=1 dual-GPU pmap" >&2
fi

# Pin one GPU per MPI rank when spatial ML is enabled (local rank -> CUDA_VISIBLE_DEVICES).
_SPATIAL_MPI=0
for ((i=1; i<=${#@}; i++)); do
  if [[ "${!i}" == "--ml-spatial-mpi" ]]; then
    _SPATIAL_MPI=1
    export MMML_MLPOT_SPATIAL_MPI=1
    break
  fi
done
if [[ "$MPI_NP" -gt 1 && ( "${MMML_MLPOT_SPATIAL_MPI:-}" == 1 || "${MMML_MLPOT_SPATIAL_MPI:-}" == true || "${MMML_MLPOT_SPATIAL_MPI:-}" == yes || "$_SPATIAL_MPI" == 1 ) ]]; then
  export MMML_MPI_PIN_GPU_PER_RANK="${MMML_MPI_PIN_GPU_PER_RANK:-1}"
fi

MMML_MPIRUN_PID=""
MMML_MPIRUN_PGID=""
MMML_MPIRUN_RC=0
MMML_MPI_CLEANUP_DONE=0
MMML_MPI_TRAP_INSTALLED=0

mmml_kill_orphan_workers() {
  local quiet_flag=()
  if [[ "${MMML_MPI_ORPHAN_CLEANUP_QUIET:-}" == 1 ]]; then
    quiet_flag=(--quiet)
  fi
  "$ROOT/scripts/mmml-kill-orphans.sh" --kill "${quiet_flag[@]}" || true
}

mmml_mpi_cleanup() {
  local reason=${1:-EXIT}
  if [[ "$MMML_MPI_CLEANUP_DONE" == 1 ]]; then
    return 0
  fi
  MMML_MPI_CLEANUP_DONE=1

  if [[ -n "$MMML_MPIRUN_PID" ]] && kill -0 "$MMML_MPIRUN_PID" 2>/dev/null; then
    echo "mmml-charmm-mpirun: stopping MPI job (${reason})..." >&2
    if [[ -n "$MMML_MPIRUN_PGID" ]]; then
      kill -TERM "-${MMML_MPIRUN_PGID}" 2>/dev/null || true
    fi
    kill -TERM "$MMML_MPIRUN_PID" 2>/dev/null || true
    local deadline=$((SECONDS + 10))
    while [[ "$SECONDS" -lt "$deadline" ]]; do
      kill -0 "$MMML_MPIRUN_PID" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$MMML_MPIRUN_PID" 2>/dev/null; then
      if [[ -n "$MMML_MPIRUN_PGID" ]]; then
        kill -KILL "-${MMML_MPIRUN_PGID}" 2>/dev/null || true
      fi
      kill -KILL "$MMML_MPIRUN_PID" 2>/dev/null || true
    fi
    wait "$MMML_MPIRUN_PID" 2>/dev/null || true
  fi
  MMML_MPIRUN_PID=""
  MMML_MPIRUN_PGID=""

  if [[ "$reason" != "EXIT" || "$MMML_MPIRUN_RC" -ne 0 ]]; then
    mmml_kill_orphan_workers
  fi
}

mmml_mpi_install_trap() {
  if [[ "$MMML_MPI_TRAP_INSTALLED" == 1 ]]; then
    return 0
  fi
  MMML_MPI_TRAP_INSTALLED=1
  trap 'mmml_mpi_cleanup INT; exit 130' INT
  trap 'mmml_mpi_cleanup TERM; exit 143' TERM
  trap 'mmml_mpi_cleanup EXIT' EXIT
}

mmml_mpi_finish() {
  local rc=$1
  local label=$2
  MMML_MPIRUN_RC=$rc
  MMML_MPI_EXIT_CODE="$rc" MMML_MPI_CRASH_ARGV0="$label" "$PY" - <<'PY' || true
import os

from mmml.interfaces.pycharmmInterface.charmm_mpi import explain_mpi_crash

explain_mpi_crash(
    int(os.environ.get("MMML_MPI_EXIT_CODE", "0")),
    argv0=os.environ.get("MMML_MPI_CRASH_ARGV0", "mmml"),
)
PY
  exit "$rc"
}

mmml_mpi_run() {
  mmml_mpi_install_trap
  local -a cmd=( "$MPIRUN" -np "$MPI_NP" "${MPIRUN_EXTRA[@]}" "$@" )
  echo "mmml-charmm-mpirun: ${cmd[*]}" >&2
  set +e
  # New session so Ctrl+C can signal the whole prterun + rank-0 tree.
  setsid "${cmd[@]}" &
  MMML_MPIRUN_PID=$!
  MMML_MPIRUN_PGID=$MMML_MPIRUN_PID
  wait "$MMML_MPIRUN_PID"
  local rc=$?
  MMML_MPIRUN_PID=""
  MMML_MPIRUN_PGID=""
  MMML_MPIRUN_RC=$rc
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
