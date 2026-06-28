#!/usr/bin/env bash
# Build libcharmm.so for GitHub Actions / clean Ubuntu (system OpenMPI 4.x).
#
# Prerequisites (apt):
#   cmake gfortran g++ libstdc++-13-dev libopenmpi-dev openmpi-bin libfftw3-dev
#
# Usage (from repo root):
#   bash scripts/ci/setup_charmm_lib.sh
#   source CHARMMSETUP
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHARMM_TAR="$ROOT/setup/charmm.tar.xz"
CHARMM_HOME="$ROOT/setup/charmm"
CHARMMSETUP="$ROOT/CHARMMSETUP"

if [[ ! -f "$CHARMM_TAR" ]]; then
  echo "ci/setup_charmm_lib: missing $CHARMM_TAR" >&2
  exit 1
fi

if [[ ! -d "$CHARMM_HOME" ]]; then
  echo "Extracting CHARMM source to $CHARMM_HOME"
  tar -xf "$CHARMM_TAR" -C "$ROOT/setup"
fi

OPENMPI_ROOT="${OPENMPI_ROOT:-/usr}"
export CHARMM_HOME CHARMM_LIB_DIR="$CHARMM_HOME" OPENMPI_ROOT
export CC="${CC:-gcc}" CXX="${CXX:-g++}"

if [[ -x "${OPENMPI_ROOT}/bin/mpicc" ]]; then
  export MPI_CC="${OPENMPI_ROOT}/bin/mpicc"
  export MPI_CXX="${OPENMPI_ROOT}/bin/mpicxx"
  export MPI_FC="${OPENMPI_ROOT}/bin/mpifort"
elif command -v mpicc >/dev/null 2>&1; then
  export MPI_CC="$(command -v mpicc)"
  export MPI_CXX="$(command -v mpicxx)"
  export MPI_FC="$(command -v mpifort)"
else
  echo "ci/setup_charmm_lib: OpenMPI wrappers not found (install libopenmpi-dev)" >&2
  exit 1
fi

# Rebuild when patches or tarball change; skip when lib already present and fresh.
PATCH_STAMP="$ROOT/setup/api/.ci-charmm-patch-stamp"
CURRENT_STAMP="$(
  sha256sum "$CHARMM_TAR" "$ROOT/setup/api/api_func.F90" "$ROOT/setup/api/api_psf.F90" \
    "$ROOT/scripts/rebuild_charmm_mlpot.sh" 2>/dev/null | sha256sum | awk '{print $1}'
)"
if [[ -f "$CHARMM_HOME/libcharmm.so" && -f "$PATCH_STAMP" && "$(cat "$PATCH_STAMP")" == "$CURRENT_STAMP" ]]; then
  echo "Reusing $CHARMM_HOME/libcharmm.so (patches unchanged)"
else
  echo "Building libcharmm.so (OpenMPI: $OPENMPI_ROOT)"
  bash "$ROOT/scripts/rebuild_charmm_mlpot.sh" --no-domdec
  mkdir -p "$(dirname "$PATCH_STAMP")"
  echo "$CURRENT_STAMP" >"$PATCH_STAMP"
fi

{
  echo "export CHARMM_HOME=$CHARMM_HOME"
  echo "export CHARMM_LIB_DIR=$CHARMM_HOME"
  echo "export OPENMPI_ROOT=$OPENMPI_ROOT"
} >"$CHARMMSETUP"

echo "Wrote optional $CHARMMSETUP (not required; mmml auto-discovers setup/charmm)"
echo "libcharmm.so: $CHARMM_HOME/libcharmm.so ($(du -h "$CHARMM_HOME/libcharmm.so" | awk '{print $1}'))"
