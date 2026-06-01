#!/usr/bin/env bash
# Launch mmml pycharmm / MLpot under OpenMPI (recommended for DOMDEC-linked libcharmm.so).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMPI_MCA_mpi_cuda_support="${OMPI_MCA_mpi_cuda_support:-0}"
export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-0}"

eval "$(python - <<'PY'
from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_library_path_export
print(mpi_library_path_export())
PY
)"

MMML="${MMML_BIN:-$ROOT/.venv/bin/mmml}"
if [[ ! -x "$MMML" ]]; then
  MMML="$(command -v mmml)"
fi

exec mpirun -np 1 "$MMML" "$@"
