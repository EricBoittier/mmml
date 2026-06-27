#!/usr/bin/env bash
# Rebuild mpi4py against the same OpenMPI as libcharmm.so (fixes MPI_Init segfaults).
#
# Symptom: backtrace shows /lib/x86_64-linux-gnu/libmpi.so inside mpi4py while
# mpirun is /opt/gcc-.../openmpi-5.0.5/build/bin/mpirun.
#
# Usage (from repo root, with CHARMM venv active):
#   export CHARMM_LIB_DIR=~/mmml/setup/charmm
#   ./scripts/rebuild_mpi4py_for_charmm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=resolve_mmml_env.sh
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

PY="${MMML_PYTHON}"
while IFS= read -r line; do
  [[ -n "$line" ]] && eval "$line"
done < <("$PY" - <<'PY'
from mmml.interfaces.pycharmmInterface.charmm_mpi import (
    charmm_mpirun_path,
    mpi4py_libmpi_path,
    charmm_libmpi_path,
    mpi_shell_setup_lines,
)
import shlex

for line in mpi_shell_setup_lines():
    print(line)

mpirun = charmm_mpirun_path()
if mpirun is None:
    raise SystemExit(
        "rebuild_mpi4py_for_charmm: no mpirun for libcharmm.so "
        "(set CHARMM_LIB_DIR / MMML_MPIRUN)"
    )
bindir = mpirun.parent
for name in ("mpicc", "mpicxx", "mpic++"):
    cc = bindir / name
    if cc.is_file():
        print(f"export MPICC={shlex.quote(str(cc))}")
        print(f"export MPICXX={shlex.quote(str(cc))}")
        break
else:
    raise SystemExit(f"rebuild_mpi4py_for_charmm: no mpicc under {bindir}")

charm = charmm_libmpi_path()
py_mpi = mpi4py_libmpi_path()
if charm is not None:
    print(f"# libcharmm libmpi: {charm}", flush=True)
if py_mpi is not None:
    print(f"# current mpi4py libmpi: {py_mpi}", flush=True)
PY
)

echo "rebuild_mpi4py_for_charmm: MPICC=${MPICC:-unset}" >&2
echo "rebuild_mpi4py_for_charmm: rebuilding mpi4py (source) into ${PY}" >&2

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" --no-binary mpi4py --force-reinstall mpi4py
else
  "$PY" -m pip install --no-binary mpi4py --force-reinstall mpi4py
fi

echo "rebuild_mpi4py_for_charmm: verify with:" >&2
echo "  mmml mpi-check" >&2
echo "  ldd \$($PY -c \"import importlib.util as u; print(u.find_spec('mpi4py.MPI').origin)\") | grep libmpi" >&2
