#!/usr/bin/env bash
# Rebuild libcharmm.so / libcharmm.dylib after changing MLpot limits in source/api/api_func.F90.
#
# Uses a local (non-NFS) build directory by default. CMake OpenMP probe compiles
# fail with "Stale file handle" when build/cmake lives on /mmhome NFS after a
# repo path move or partial cache cleanup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARMM_HOME="${CHARMM_HOME:-$ROOT/setup/charmm}"
CHARMM_TAR="${CHARMM_TAR:-$ROOT/setup/charmm.tar.xz}"
NFS_BUILD="$CHARMM_HOME/build/cmake"

_platform_tag() {
  local os arch
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"
  case "$os" in
    darwin) echo "darwin-${arch}" ;;
    linux) echo "linux-${arch}" ;;
    *) echo "${os}-${arch}" ;;
  esac
}

_default_openmpi_root() {
  if [[ -d /opt/gcc-14.2.0/openmpi-5.0.5/build/bin ]]; then
    echo /opt/gcc-14.2.0/openmpi-5.0.5/build
  elif [[ "$(uname -s)" == "Darwin" && -d /opt/homebrew/opt/open-mpi/bin ]]; then
    echo /opt/homebrew/opt/open-mpi
  elif [[ "$(uname -s)" == "Darwin" && -d /opt/homebrew/bin ]]; then
    echo /opt/homebrew
  elif [[ -d /usr/bin/mpicc ]]; then
    echo /usr
  else
    echo /usr
  fi
}

_auto_find_fftw_root() {
  # Find an FFTW prefix that has fftw3.h AND libfftw3f (single-precision, required for COLFFT).
  # Tries bare prefix first (e.g. /srv/opt/gcc-12.2.0/fftw-3.3.10), then /build sub-dir.
  local candidate base
  for pattern in \
    "/srv/opt/gcc-*/fftw-*" \
    "/srv/opt/fftw-*" \
    "/opt/gcc-*/fftw-*" \
    "/opt/fftw-*" \
    "/usr/local" \
    "/usr"; do
    # shellcheck disable=SC2086
    for base in $(ls -d $pattern 2>/dev/null | sort -rV 2>/dev/null | head -5); do
      for candidate in "$base" "$base/build"; do
        if [[ -f "$candidate/include/fftw3.h" ]] && \
           { [[ -f "$candidate/lib/libfftw3f.a" ]] || \
             [[ -f "$candidate/lib/libfftw3f.so" ]] || \
             [[ -f "$candidate/lib64/libfftw3f.a" ]] || \
             [[ -f "$candidate/lib64/libfftw3f.so" ]]; }; then
          echo "$candidate"
          return 0
        fi
      done
    done
  done
  # pkg-config fallback
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists fftw3f 2>/dev/null; then
    local inc
    inc="$(pkg-config --variable=includedir fftw3f 2>/dev/null || true)"
    [[ -n "$inc" && -f "$inc/fftw3.h" ]] && { echo "$(dirname "$inc")"; return 0; }
  fi
  return 1
}

PLATFORM_TAG="$(_platform_tag)"
LOCAL_BUILD="${CHARMM_BUILD_DIR:-${HOME}/.cache/mmml-charmm-build/${PLATFORM_TAG}}"
OPENMPI_ROOT="${OPENMPI_ROOT:-$(_default_openmpi_root)}"
PMIX_LIB="${PMIX_LIB:-/opt/gcc-14.2.0/pmix-5.0.4/lib}"
# FFTW_ROOT resolution order:
#  1. Explicit env var FFTW_ROOT
#  2. EBROOTFFTW set by Environment Modules (module load FFTW/...)
#  3. Auto-discovery under /srv/opt/gcc-*/fftw-*/build and similar prefixes
FFTW_ROOT="${FFTW_ROOT:-${EBROOTFFTW:-}}"
if [[ -z "$FFTW_ROOT" ]]; then
  if _found="$(_auto_find_fftw_root 2>/dev/null)"; then
    FFTW_ROOT="$_found"
    echo "FFTW_ROOT auto-detected: $FFTW_ROOT" >&2
  fi
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  LIB_BASENAME="libcharmm.dylib"
else
  LIB_BASENAME="libcharmm.so"
fi
LIB_OUT="$CHARMM_HOME/$LIB_BASENAME"

CLEAN=0
USE_NFS_BUILD=0
DEBUG=0
SYNC_PATCHES=1
NO_DOMDEC=0
SKIP_PACKMOL=0
NATIVE_EXEC=0
CHARMM_BUILD_TYPE="${CHARMM_BUILD_TYPE:-Release}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--clean] [--use-nfs-build] [--debug] [--no-sync-patches] [--no-domdec] [--skip-packmol] [--native-exec]

  --clean             Remove the cmake build directory and reconfigure from scratch.
  --use-nfs-build     Build in setup/charmm/build/cmake (default: \$HOME/.cache/mmml-charmm-build/<platform>).
  --debug             RelWithDebInfo + -g -fbacktrace (readable gdb/addr2line on segfaults).
  --no-sync-patches   Skip copying setup/api/api_func.F90 into the CHARMM tree.
  --no-domdec         CMake -Ddomdec=OFF (no DOMDEC send_coord_to_recip path; MPI MLpot SD).
  --skip-packmol      Skip rebuilding mmml/generate/packmol/packmol for this platform.
  --native-exec       Build charmm executable (as_library=OFF) for DOMDEC tier3 smoke; skips Packmol.

Build profile (default): MPI + DOMDEC + COLFFT + KEY_LIBRARY (as_library=ON), domdec_gpu=OFF.
Also builds Packmol (mmml/generate/packmol/packmol) unless --skip-packmol or --native-exec.
MLpot workflows run with DOMDEC compiled in but disabled at runtime (domdec off, mpirun -np 1).
Use --no-domdec when MLpot SD still segfaults in send_coord_to_recip after JAX warmup.
Use --native-exec (or scripts/rebuild_charmm_native_exec.sh) for Tier 3 DOMDEC np>1 smoke.

Environment:
  CHARMM_HOME       CHARMM source tree (default: $ROOT/setup/charmm)
  CHARMM_BUILD_DIR  CMake build directory (default: \$HOME/.cache/mmml-charmm-build/${PLATFORM_TAG})
  CHARMM_BUILD_TYPE CMake build type (default: Release; --debug sets RelWithDebInfo)
  OPENMPI_ROOT      OpenMPI prefix (default: auto-detect; Linux cluster: /opt/gcc-14.2.0/openmpi-5.0.5/build)
  FFTW_ROOT         FFTW prefix; falls back to EBROOTFFTW when modules set it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --use-nfs-build) USE_NFS_BUILD=1; shift ;;
    --debug) DEBUG=1; CHARMM_BUILD_TYPE=RelWithDebInfo; shift ;;
    --no-sync-patches) SYNC_PATCHES=0; shift ;;
    --no-domdec) NO_DOMDEC=1; shift ;;
    --skip-packmol) SKIP_PACKMOL=1; shift ;;
    --native-exec) NATIVE_EXEC=1; SKIP_PACKMOL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ensure_charmm_cmake_source() {
  if [[ -f "$CHARMM_HOME/CMakeLists.txt" ]]; then
    return 0
  fi
  if [[ ! -f "$CHARMM_TAR" ]]; then
    echo "rebuild_charmm_mlpot: missing $CHARMM_HOME/CMakeLists.txt and $CHARMM_TAR" >&2
    echo "Extract setup/charmm from setup/charmm.tar.xz (bash setup/install.sh) or restore CMakeLists.txt." >&2
    exit 1
  fi
  echo "Extracting CMakeLists.txt and tool/cmake from $CHARMM_TAR"
  tar -xf "$CHARMM_TAR" -C "$ROOT/setup" charmm/CMakeLists.txt charmm/tool/cmake
}

ensure_charmm_cmake_source

F90="$CHARMM_HOME/source/api/api_func.F90"
PATCH_F90="${MMML_PATCH_SOURCE:-$ROOT/setup/api/api_func.F90}"
if [[ ! -f "$F90" ]]; then
  echo "rebuild_charmm_mlpot: missing $F90" >&2
  exit 1
fi

if [[ "$SYNC_PATCHES" == 1 && -f "$PATCH_F90" ]]; then
  if ! cmp -s "$PATCH_F90" "$F90"; then
    echo "Syncing MLpot limits from $PATCH_F90"
    cp -f "$PATCH_F90" "$F90"
  fi
elif [[ "$SYNC_PATCHES" == 1 ]]; then
  echo "rebuild_charmm_mlpot: patch source not found: $PATCH_F90" >&2
  exit 1
fi

PSF_F90="$CHARMM_HOME/source/api/api_psf.F90"
PATCH_PSF_F90="$ROOT/setup/api/api_psf.F90"
if [[ ! -f "$PSF_F90" ]]; then
  echo "rebuild_charmm_mlpot: missing $PSF_F90" >&2
  exit 1
fi

if [[ "$SYNC_PATCHES" == 1 && -f "$PATCH_PSF_F90" ]]; then
  if ! cmp -s "$PATCH_PSF_F90" "$PSF_F90"; then
    echo "Syncing api_psf.F90 from $PATCH_PSF_F90"
    cp -f "$PATCH_PSF_F90" "$PSF_F90"
  fi
fi

echo "MLpot limits in source:"
grep -E 'max_Nml|max_Npr' "$F90" || true

# Match OpenMPI used by the installed libcharmm (not an unrelated system OpenMPI).
if [[ -d "$OPENMPI_ROOT/bin" ]]; then
  export PATH="$OPENMPI_ROOT/bin:${PATH}"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="${OPENMPI_ROOT}/lib:${PMIX_LIB}:${DYLD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="${OPENMPI_ROOT}/lib:${PMIX_LIB}:${LD_LIBRARY_PATH:-}"
  fi
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
# mpi_shell_setup_lines prepends /usr/lib64 before the appended gcc libstdc++ path;
# cmake from /opt/gcc-12.2.0 then fails with GLIBCXX_3.4.30 not found.
for _libdir in /opt/gcc-12.2.0/build/lib64 /opt/gcc-12.2.0/build/lib/gcc/x86_64-pc-linux-gnu/12.2.0; do
  if [[ -d "$_libdir" ]]; then
    export LD_LIBRARY_PATH="${_libdir}:${LD_LIBRARY_PATH:-}"
  fi
done

BUILD_DIR="$LOCAL_BUILD"
if [[ "$NATIVE_EXEC" == 1 && -z "${CHARMM_BUILD_DIR:-}" ]]; then
  BUILD_DIR="${LOCAL_BUILD}-exec"
fi
if [[ "$USE_NFS_BUILD" == 1 ]]; then
  BUILD_DIR="$NFS_BUILD"
  if [[ "$NATIVE_EXEC" == 1 && -z "${CHARMM_BUILD_DIR:-}" ]]; then
    BUILD_DIR="${NFS_BUILD}-exec"
  fi
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

# ---------------------------------------------------------------------------
# Function must be defined BEFORE the configure block that calls it.
# (bash resolves function names at call time within a running block, so a
#  definition that appears after the call site is not yet in scope.)
# ---------------------------------------------------------------------------
_assert_charmm_domdec_cmake_flags() {
  [[ -f "$BUILD_DIR/CMakeCache.txt" ]] || return 0
  if [[ "$NO_DOMDEC" == 1 ]]; then
    return 0
  fi
  local domdec_val colfft_val
  domdec_val="$(grep '^domdec:BOOL=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2- || true)"
  colfft_val="$(grep '^colfft:BOOL=' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | cut -d= -f2- || true)"
  if [[ "$domdec_val" == "ON" && "$colfft_val" == "ON" ]]; then
    echo "CMake cache: domdec=ON colfft=ON"
    return 0
  fi
  echo "rebuild_charmm_mlpot: ERROR: DOMDEC requires colfft=ON; CMake has domdec=${domdec_val:-?} colfft=${colfft_val:-?}" >&2
  echo "CHARMM CMake turns domdec OFF when colfft is OFF (no FFTW/MKL found)." >&2
  echo "" >&2
  # Try auto-discovery and print a ready-to-paste export if found
  local _auto_fftw
  if _auto_fftw="$(_auto_find_fftw_root 2>/dev/null)"; then
    echo "Auto-detected FFTW at: $_auto_fftw" >&2
    echo "  export FFTW_ROOT=$_auto_fftw" >&2
    echo "  bash scripts/rebuild_charmm_native_exec.sh --clean" >&2
  else
    echo "Could not auto-detect FFTW. Try:" >&2
    echo "  module avail 2>&1 | grep -i fftw     # list available FFTW modules" >&2
    echo "  module load <fftw-module-name>        # load it" >&2
    echo "  export FFTW_ROOT=\${EBROOTFFTW}        # set by the module" >&2
    echo "" >&2
    echo "Or set it manually (check /srv/opt/gcc-*/fftw-*/build):" >&2
    echo "  ls /srv/opt/gcc-12.2.0/ 2>/dev/null | grep fftw" >&2
    echo "  export FFTW_ROOT=/srv/opt/gcc-12.2.0/fftw-<ver>/build" >&2
    echo "" >&2
    echo "Then: bash scripts/rebuild_charmm_native_exec.sh --clean" >&2
  fi
  exit 1
}

# ---------------------------------------------------------------------------
# FFTW auto-discovery: if FFTW_ROOT is not set, try to recover it from the
# library build's CMakeCache (which succeeded when the library was built).
# ---------------------------------------------------------------------------
if [[ -z "$FFTW_ROOT" ]]; then
  _lib_cache="${HOME}/.cache/mmml-charmm-build/$(_platform_tag)/CMakeCache.txt"
  if [[ -f "$_lib_cache" ]]; then
    _fftw_inc="$(grep '^FFTW_INCLUDE_DIR:PATH=' "$_lib_cache" 2>/dev/null | cut -d= -f2- || true)"
    if [[ -n "$_fftw_inc" && -d "$_fftw_inc" ]]; then
      FFTW_ROOT="$(dirname "$_fftw_inc")"
      echo "FFTW_ROOT auto-discovered from library cache: $FFTW_ROOT (include: $_fftw_inc)" >&2
    fi
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

# Large max_Npr tiers allocate multi-GB static arrays in api_func.F90 (.bss). On
# x86_64 the default small code model cannot link libcharmm.so: R_X86_64_PC32
# relocation truncated to fit against .bss (idxi/idxj and other module symbols).
CODE_MODEL_FLAG=""
if [[ "$(uname -m)" == "x86_64" ]]; then
  CODE_MODEL_FLAG="-mcmodel=medium"
fi

if [[ "$needs_configure" == 0 && -n "$CODE_MODEL_FLAG" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cached_fflags="$(grep '^CMAKE_Fortran_FLAGS:STRING=' "$BUILD_DIR/CMakeCache.txt" | cut -d= -f2- || true)"
  if [[ "$cached_fflags" != *mcmodel=medium* ]]; then
    echo "CMake cache lacks -mcmodel=medium; reconfiguring in $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    needs_configure=1
  fi
fi

if [[ "$needs_configure" == 1 ]]; then
  mkdir -p "$BUILD_DIR"
  _as_library=ON
  [[ "$NATIVE_EXEC" == 1 ]] && _as_library=OFF
  echo "Configuring CHARMM in $BUILD_DIR (OpenMPI: $OPENMPI_ROOT, build: $CHARMM_BUILD_TYPE, as_library=${_as_library}) ..."
  CMAKE_ARGS=(
    -S "$CHARMM_HOME"
    -B "$BUILD_DIR"
    -DCMAKE_INSTALL_PREFIX="$CHARMM_HOME"
    -DCMAKE_BUILD_TYPE="$CHARMM_BUILD_TYPE"
    -Das_library="${_as_library}"
    -Din_place_install=ON
    -Dopenmm=OFF
    -Dcolfft=ON
    -Ddomdec="$([[ "$NO_DOMDEC" == 1 ]] && echo OFF || echo ON)"
    -DMPI_C_COMPILER="$MPI_CC"
    -DMPI_CXX_COMPILER="$MPI_CXX"
    -DMPI_Fortran_COMPILER="$MPI_FC"
  )
  if [[ "$(uname -s)" == "Darwin" ]]; then
    CMAKE_ARGS+=(
      -Dcuda=OFF
      -Ddomdec_gpu=OFF
      -Dopencl=OFF
      -Dqchem=OFF
    )
  fi
  FFLAGS="$CODE_MODEL_FLAG"
  CFLAGS="$CODE_MODEL_FLAG"
  CXXFLAGS="$CODE_MODEL_FLAG"
  if [[ "$DEBUG" == 1 ]]; then
    FFLAGS+=" -g -fbacktrace -fno-omit-frame-pointer"
    CFLAGS+=" -g -fno-omit-frame-pointer"
    CXXFLAGS+=" -g -fno-omit-frame-pointer"
  fi
  if [[ -n "$FFLAGS" ]]; then
    CMAKE_ARGS+=(
      -DCMAKE_Fortran_FLAGS="$FFLAGS"
      -DCMAKE_C_FLAGS="$CFLAGS"
      -DCMAKE_CXX_FLAGS="$CXXFLAGS"
      -DCMAKE_SHARED_LINKER_FLAGS="$CODE_MODEL_FLAG"
    )
    echo "Using code model flags: $FFLAGS (linker: $CODE_MODEL_FLAG)"
  fi
  if [[ -n "$FFTW_ROOT" ]]; then
    FFTW_LIB_DIR=""
    for candidate in "$FFTW_ROOT/lib" "$FFTW_ROOT/lib64"; do
      if [[ -d "$candidate" ]]; then
        FFTW_LIB_DIR="$candidate"
        break
      fi
    done
    if [[ -n "$FFTW_LIB_DIR" ]]; then
      FFTW_LIB=""    # double precision (libfftw3)
      FFTWF_LIB=""   # single precision (libfftw3f) — required for COLFFT/PME
      # Prefer shared libs (.so/.dylib); fall back to static (.a) — CMake needs an
      # explicit path when the install only provides static archives.
      for candidate in \
          "$FFTW_LIB_DIR"/libfftw3.so "$FFTW_LIB_DIR"/libfftw3.so.* \
          "$FFTW_LIB_DIR"/libfftw3.dylib \
          "$FFTW_LIB_DIR"/libfftw3.a; do
        if [[ -f "$candidate" ]]; then FFTW_LIB="$candidate"; break; fi
      done
      for candidate in \
          "$FFTW_LIB_DIR"/libfftw3f.so "$FFTW_LIB_DIR"/libfftw3f.so.* \
          "$FFTW_LIB_DIR"/libfftw3f.dylib \
          "$FFTW_LIB_DIR"/libfftw3f.a; do
        if [[ -f "$candidate" ]]; then FFTWF_LIB="$candidate"; break; fi
      done
      CMAKE_ARGS+=(
        -DFFTW_ROOT="$FFTW_ROOT"
        -DFFTW_INCLUDE_DIR="$FFTW_ROOT/include"
        -DFFTW_INCLUDE_DIRS="$FFTW_ROOT/include"
      )
      if [[ -n "$FFTW_LIB" ]]; then
        CMAKE_ARGS+=(-DFFTW_LIBRARY="$FFTW_LIB" -DFFTW_LIBRARIES="$FFTW_LIB")
        echo "Using FFTW (double) from $FFTW_ROOT ($FFTW_LIB)"
      fi
      if [[ -n "$FFTWF_LIB" ]]; then
        CMAKE_ARGS+=(-DFFTWF_LIBRARY="$FFTWF_LIB" -DFFTWF_LIBRARIES="$FFTWF_LIB")
        echo "Using FFTWF (single, required for COLFFT) from $FFTW_ROOT ($FFTWF_LIB)"
      else
        echo "rebuild_charmm_mlpot: warning: libfftw3f not found under $FFTW_LIB_DIR" >&2
        echo "  COLFFT/DOMDEC requires single-precision FFTW (libfftw3f)." >&2
        if [[ -d "$(dirname "$FFTW_ROOT")/fftw-3.3.10-dp" ]]; then
          echo "  Found fftw-3.3.10-dp sibling — check if it has libfftw3f." >&2
        fi
      fi
    else
      echo "rebuild_charmm_mlpot: warning: FFTW_ROOT=$FFTW_ROOT has no lib or lib64 directory" >&2
    fi
  fi
  cmake "${CMAKE_ARGS[@]}"
  _assert_charmm_domdec_cmake_flags
fi

# Re-check an existing cache before native-exec builds (configure may have been skipped).
if [[ "$needs_configure" == 0 && "$NO_DOMDEC" != 1 ]]; then
  _assert_charmm_domdec_cmake_flags
fi

_build_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

if [[ "$NATIVE_EXEC" == 1 ]]; then
  EXE_OUT="$CHARMM_HOME/charmm"
  echo "Building charmm executable in $BUILD_DIR ..."
  cmake --build "$BUILD_DIR" -j "$(_build_jobs)"
  cmake --install "$BUILD_DIR" || true

  BUILT=""
  for candidate in \
    "$CHARMM_HOME/charmm" \
    "$CHARMM_HOME/bin/charmm" \
    "$CHARMM_HOME/exec/charmm" \
    "$BUILD_DIR/charmm" \
    "$BUILD_DIR/bin/charmm" \
    "$BUILD_DIR/exec/charmm"; do
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      BUILT="$candidate"
      break
    fi
  done
  if [[ -z "$BUILT" ]]; then
    BUILT="$(find "$BUILD_DIR" -type f -name charmm -perm -111 -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "$BUILT" ]]; then
    echo "rebuild_charmm_mlpot: build finished but charmm executable not found under $BUILD_DIR" >&2
    exit 1
  fi

  _real_built="$(readlink -f "$BUILT" 2>/dev/null || echo "$BUILT")"
  _real_out="$(readlink -f "$EXE_OUT" 2>/dev/null || echo "$EXE_OUT")"
  if [[ "$_real_built" != "$_real_out" ]]; then
    cp -f "$BUILT" "$EXE_OUT"
  fi
  chmod +x "$EXE_OUT"
  echo "Installed $EXE_OUT (from $BUILT)"
  if [[ "$DEBUG" == 1 ]]; then
    if command -v readelf >/dev/null 2>&1 && readelf -S "$EXE_OUT" 2>/dev/null | grep -q '\.debug'; then
      echo "Debug symbols: present in $EXE_OUT"
    elif [[ "$(uname -s)" == "Darwin" ]] && dsymutil "$EXE_OUT" >/dev/null 2>&1; then
      echo "Debug symbols: present in $EXE_OUT (Darwin dSYM)"
    else
      echo "rebuild_charmm_mlpot: warning: no debug sections in $EXE_OUT (gdb backtraces may lack line numbers)" >&2
    fi
  fi
  cat <<EOF
Verify Tier 3 DOMDEC smoke (dense ~40Å prep, np=2):
  CHARMM_EXE=$EXE_OUT bash scripts/run_domdec_dcm10_smoke.sh tier3

If domdec ndir is "extraneous" at runtime, DOMDEC was not compiled in (check domdec/colfft in CMakeCache).
Site c47 /opt/charmm/c47* rejects np=2 NDIR; use this MMML native binary instead.
EOF
  exit 0
fi

echo "Building $LIB_BASENAME in $BUILD_DIR ..."
cmake --build "$BUILD_DIR" -j "$(_build_jobs)"
cmake --install "$BUILD_DIR" || true

BUILT=""
for candidate in \
  "$CHARMM_HOME/lib/$LIB_BASENAME" \
  "$CHARMM_HOME/$LIB_BASENAME" \
  "$BUILD_DIR/$LIB_BASENAME" \
  "$BUILD_DIR/lib/$LIB_BASENAME"; do
  if [[ -f "$candidate" ]]; then
    BUILT="$candidate"
    break
  fi
done
if [[ -z "$BUILT" ]]; then
  BUILT="$(find "$BUILD_DIR" -name "$LIB_BASENAME" -print -quit || true)"
fi
if [[ -z "$BUILT" ]]; then
  echo "rebuild_charmm_mlpot: build finished but $LIB_BASENAME not found" >&2
  exit 1
fi

cp -f "$BUILT" "$LIB_OUT"
echo "Installed $LIB_OUT (from $BUILT)"
if [[ "$SKIP_PACKMOL" != 1 ]]; then
  PACKMOL_ARGS=()
  [[ "$CLEAN" == 1 ]] && PACKMOL_ARGS+=(--clean)
  bash "$ROOT/scripts/rebuild_packmol.sh" "${PACKMOL_ARGS[@]}"
fi
if [[ "$DEBUG" == 1 ]]; then
  if command -v readelf >/dev/null 2>&1 && readelf -S "$LIB_OUT" 2>/dev/null | grep -q '\.debug'; then
    echo "Debug symbols: present in $LIB_OUT"
  elif [[ "$(uname -s)" == "Darwin" ]] && dsymutil "$LIB_OUT" >/dev/null 2>&1; then
    echo "Debug symbols: present in $LIB_OUT (Darwin dSYM)"
  else
    echo "rebuild_charmm_mlpot: warning: no debug sections in $LIB_OUT (gdb backtraces may lack line numbers)" >&2
  fi
fi
cat <<EOF
Verify:
  uv run python -c "from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import mlpot_limits_message; print(mlpot_limits_message())"
Expect: max_Nml=50000, max_Npr=3998000, source=api_func.F90 ($LIB_BASENAME is up to date)
  uv run python -c "from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable; print(packmol_executable())"
If you see max_Nml=100: rebuild with scripts/rebuild_charmm_mlpot.sh (paths auto-discover from setup/charmm).

Segfault diagnosis:
  MMML_MPI_GDB=1 ./scripts/mmml-charmm-mpirun.sh md-system --config ...
  (uses OMPI_MCA_orte_abort_print_stack=1 by default; ignore PRRTE Sphinx help noise)
EOF
