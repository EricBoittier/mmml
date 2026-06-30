#!/usr/bin/env bash
# Minimal PyCHARMM np>1 cooperative READ gate (cluster manual bisect).
#
# Usage:
#   MMML_MPI_NP=1 ./scripts/run_mpi_pycharmm_read_gate.sh
#   MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode psf-crd
#   MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode stream-inp
#   MMML_MPI_NP=4 ./scripts/run_mpi_pycharmm_read_gate.sh --mode restart --with-crystal
#
# Requires prebuilt artifacts from:
#   MMML_MPI_NP=1 ./scripts/mmml-charmm-mpirun.sh python \
#     tests/functionality/mlpot/10_domdec_spatial_mpi_smoke.py \
#     --prepare-prebuilt-only --residue DCM --n-molecules 20 --box-side 40

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMML_MPI_NP="${MMML_MPI_NP:-1}"
export MMML_MPI_NP

# np=1 serial validation: skip mpirun.  MPI-linked libcharmm.so + mpirun -np 1
# hangs on reset_block and leaves eval_charmm_script READ with n_atoms=0; native
# -i and plain Python (no mpirun) do not hit that path.
if [[ "$MMML_MPI_NP" == 1 ]]; then
  # shellcheck source=resolve_mmml_env.sh
  source "$ROOT/scripts/resolve_mmml_env.sh"
  mmml_resolve_env "$ROOT"
  PY="${MMML_PYTHON:?MMML_PYTHON unset after resolve_mmml_env}"
  exec "$PY" "$ROOT/tests/functionality/charmm/mpi_pycharmm_read_gate.py" "$@"
fi

exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python \
  "$ROOT/tests/functionality/charmm/mpi_pycharmm_read_gate.py" "$@"
