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

exec "$ROOT/scripts/mmml-charmm-mpirun.sh" python \
  "$ROOT/tests/functionality/charmm/mpi_pycharmm_read_gate.py" "$@"
