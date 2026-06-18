#!/usr/bin/env bash
# Rebuild libcharmm.so after changing MLpot limits in source/api/api_func.F90.
#
# Uses a local (non-NFS) build directory by default. CMake OpenMP probe compiles
# fail with "Stale file handle" when build/cmake lives on /mmhome NFS after a
# repo path move or partial cache cleanup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARMM_HOME="${CHARMM_HOME:-$ROOT/setup/charmm}"
LIB_OUT="$CHARMM_HOME/libcharmm.so"
NFS_BUILD="$CHARMM_HOME/build/cmake"
LOCAL_BUILD="${CHARMM_BUILD_DIR:-${HOME}/.cache/mmml-charmm-build}"
OPENMPI_ROOT="${OPENMPI_ROOT:-/opt/gcc-14.2.0/openmpi-5.0.5/build}"
PMIX_LIB="${PMIX_LIB:-/opt/gcc-14.2.0/pmix-5.0.4/lib}"

CLEAN=0
USE_NFS_BUILD=0
DEBUG=0
CHARMM_BUILD_TYPE="${CHARMM_BUILD_TYPE:-Release}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--clean] [--use-nfs-build] [--debug]

  --clean           Remove the cmake build directory and reconfigure from scratch.
  --use-nfs-build   Build in setup/charmm/build/cmake (default: \$HOME/.cache/mmml-charmm-build).
  --debug           RelWithDebInfo + -g -fbacktrace (readable gdb/addr2line on segfaults).

Environment:
  CHARMM_HOME       CHARMM source tree (default: $ROOT/setup/charmm)
  CHARMM_BUILD_DIR  CMake build directory (default: \$HOME/.cache/mmml-charmm-build)
  CHARMM_BUILD_TYPE CMake build type (default: Release; --debug sets RelWithDebInfo)
  OPENMPI_ROOT      OpenMPI 5 prefix (default: $OPENMPI_ROOT)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --use-nfs-build) USE_NFS_BUILD=1; shift ;;
    --debug) DEBUG=1; CHARMM_BUILD_TYPE=RelWithDebInfo; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

F90="$CHARMM_HOME/source/api/api_func.F90"
if [[ ! -f "$F90" ]]; then
  echo "rebuild_charmm_mlpot: missing $F90" >&2
  exit 1
fi

echo "MLpot limits in source:"
grep -E 'max_Nml|max_Npr' "$F90" || true

# Match OpenMPI 5 used by the installed libcharmm.so (not system OpenMPI 3).
if [[ -d "$OPENMPI_ROOT/bin" ]]; then
  export PATH="$OPENMPI_ROOT/bin:${PATH}"
  export LD_LIBRARY_PATH="${OPENMPI_ROOT}/lib:${PMIX_LIB}:${LD_LIBRARY_PATH:-}"
fi
if command -v python3 >/dev/null 2>&1; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && eval "$line"
  done < <(
    python3 -c "
from mmml.interfaces.pycharmmInterface.charmm_mpi import mpi_shell_setup_lines
print('\n'.join(mpi_shell_setup_lines()))
" 2>/dev/null || true
  )
fi

BUILD_DIR="$LOCAL_BUILD"
if [[ "$USE_NFS_BUILD" == 1 ]]; then
  BUILD_DIR="$NFS_BUILD"
fi

# Drop stale NFS cmake cache from an old repo path (e.g. studixh -> mmhome).
if [[ -d "$NFS_BUILD" ]]; then
  if [[ -f "$NFS_BUILD/CMakeCache.txt" ]]; then
    if grep -qE 'studixh|CMAKE_HOME_DIRECTORY:INTERNAL=.*/mmml/setup/charmm/build/cmake' \
      "$NFS_BUILD/CMakeCache.txt" 2>/dev/null; then
      cached_src="$(grep '^CHARMM_SOURCE_DIR:STATIC=' "$NFS_BUILD/CMakeCache.txt" | cut -d= -f2- || true)"
      if [[ -n "$cached_src" && "$cached_src" != "$CHARMM_HOME" ]]; then
        echo "Removing stale NFS cmake cache at $NFS_BUILD (was $cached_src)"
        rm -rf "$NFS_BUILD"
      fi
    fi
  fi
fi

needs_configure=0
if [[ "$CLEAN" == 1 ]]; then
  echo "Cleaning $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  needs_configure=1
elif [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  needs_configure=1
else
  cached_src="$(grep '^CHARMM_SOURCE_DIR:STATIC=' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ -n "$cached_src" && "$cached_src" != "$CHARMM_HOME" ]]; then
    echo "Stale cmake cache (source was $cached_src); reconfiguring in $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    needs_configure=1
  fi
fi

MPI_CC="${MPI_CC:-${OPENMPI_ROOT}/bin/mpicc}"
MPI_CXX="${MPI_CXX:-${OPENMPI_ROOT}/bin/mpicxx}"
MPI_FC="${MPI_FC:-${OPENMPI_ROOT}/bin/mpifort}"
for wrapper in "$MPI_CC" "$MPI_CXX" "$MPI_FC"; do
  if [[ ! -x "$wrapper" ]]; then
    echo "rebuild_charmm_mlpot: missing MPI wrapper $wrapper" >&2
    echo "Set OPENMPI_ROOT or MPI_CC/MPI_CXX/MPI_FC and retry." >&2
    exit 1
  fi
done

if [[ "$needs_configure" == 1 ]]; then
  mkdir -p "$BUILD_DIR"
  echo "Configuring CHARMM library in $BUILD_DIR (OpenMPI: $OPENMPI_ROOT, build: $CHARMM_BUILD_TYPE) ..."
  CMAKE_ARGS=(
    -S "$CHARMM_HOME"
    -B "$BUILD_DIR"
    -DCMAKE_INSTALL_PREFIX="$CHARMM_HOME"
    -DCMAKE_BUILD_TYPE="$CHARMM_BUILD_TYPE"
    -Das_library=ON
    -Din_place_install=ON
    -Dopenmm=OFF
    -DMPI_C_COMPILER="$MPI_CC"
    -DMPI_CXX_COMPILER="$MPI_CXX"
    -DMPI_Fortran_COMPILER="$MPI_FC"
  )
  if [[ "$DEBUG" == 1 ]]; then
    CMAKE_ARGS+=(
      -DCMAKE_Fortran_FLAGS="-g -fbacktrace -fno-omit-frame-pointer"
      -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer"
      -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer"
    )
  fi
  cmake "${CMAKE_ARGS[@]}"
fi

echo "Building libcharmm.so in $BUILD_DIR ..."
cmake --build "$BUILD_DIR" -j "$(nproc)"
cmake --install "$BUILD_DIR"

BUILT=""
for candidate in \
  "$CHARMM_HOME/lib/libcharmm.so" \
  "$BUILD_DIR/libcharmm.so" \
  "$BUILD_DIR/lib/libcharmm.so"; do
  if [[ -f "$candidate" ]]; then
    BUILT="$candidate"
    break
  fi
done
if [[ -z "$BUILT" ]]; then
  BUILT="$(find "$BUILD_DIR" -name 'libcharmm.so' -print -quit || true)"
fi
if [[ -z "$BUILT" ]]; then
  echo "rebuild_charmm_mlpot: build finished but libcharmm.so not found" >&2
  exit 1
fi

cp -f "$BUILT" "$LIB_OUT"
echo "Installed $LIB_OUT (from $BUILT)"
if [[ "$DEBUG" == 1 ]]; then
  if command -v readelf >/dev/null 2>&1 && readelf -S "$LIB_OUT" 2>/dev/null | grep -q '\.debug'; then
    echo "Debug symbols: present in $LIB_OUT"
  else
    echo "rebuild_charmm_mlpot: warning: no .debug sections in $LIB_OUT (gdb backtraces may lack line numbers)" >&2
  fi
fi
cat <<EOF
Verify:
  uv run python -c "from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import mlpot_limits_message; print(mlpot_limits_message())"
Expect: max_Nml=50000, max_Npr=3998000, source=api_func.F90 (libcharmm.so is up to date)
If you see max_Nml=100: set CHARMM_HOME/CHARMM_LIB_DIR in CHARMMSETUP or export them, then rebuild again.

Segfault diagnosis:
  MMML_MPI_GDB=1 ./scripts/mmml-charmm-mpirun.sh md-system --config ...
  (uses OMPI_MCA_orte_abort_print_stack=1 by default; ignore PRRTE Sphinx help noise)
EOF
