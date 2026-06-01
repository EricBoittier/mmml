#!/usr/bin/env bash
# Launch mmml pycharmm / MLpot under the OpenMPI build linked to libcharmm.so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while IFS= read -r line; do
  [[ -n "$line" ]] && eval "$line"
done < <(python - <<'PY'
from mmml.interfaces.pycharmmInterface.charmm_mpi import charmm_mpirun_path, mpi_shell_setup_lines
for line in mpi_shell_setup_lines():
    print(line)
mpirun = charmm_mpirun_path()
if mpirun is None:
    raise SystemExit(
        "mmml-charmm-mpirun: could not find OpenMPI mpirun for libcharmm.so. "
        "Set MMML_MPIRUN=/path/to/mpirun or MMML_MPI_LD_PATH_EXTRA."
    )
PY
)

MMML="${MMML_BIN:-$ROOT/.venv/bin/mmml}"
if [[ ! -x "$MMML" ]]; then
  MMML="$(command -v mmml)"
fi

MPIRUN="${MMML_MPIRUN:-mpirun}"
exec "$MPIRUN" -np 1 "$MMML" "$@"
