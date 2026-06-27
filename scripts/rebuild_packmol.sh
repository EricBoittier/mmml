#!/usr/bin/env bash
# Build the vendored Packmol binary for the current platform.
#
# Installs to mmml/generate/packmol/packmol (used by packmol_placement / make-box).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKMOL_SRC="${PACKMOL_SRC:-$ROOT/mmml/generate/packmol}"
PACKMOL_OUT="${PACKMOL_OUT:-$PACKMOL_SRC/packmol}"

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

_build_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

_binary_runs_on_host() {
  local bin="$1"
  [[ -f "$bin" && -x "$bin" ]] || return 1
  if ! command -v file >/dev/null 2>&1; then
    return 0
  fi
  local desc
  desc="$(file -b "$bin" 2>/dev/null || true)"
  case "$(uname -s)" in
    Darwin)
      [[ "$desc" == *"Mach-O"* ]]
      ;;
    Linux)
      [[ "$desc" == *"ELF"* ]]
      ;;
    *)
      return 0
      ;;
  esac
}

CLEAN=0
FORCE=0
PLATFORM_TAG="$(_platform_tag)"
BUILD_DIR="${PACKMOL_BUILD_DIR:-${HOME}/.cache/mmml-packmol-build/${PLATFORM_TAG}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--clean] [--force]

Build Packmol from mmml/generate/packmol and install to:
  $PACKMOL_OUT

Environment:
  PACKMOL_SRC       Packmol source tree (default: $PACKMOL_SRC)
  PACKMOL_OUT       Output executable path (default: $PACKMOL_OUT)
  PACKMOL_BUILD_DIR CMake build directory (default: \$HOME/.cache/mmml-packmol-build/${PLATFORM_TAG})
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$PACKMOL_SRC/CMakeLists.txt" ]]; then
  echo "rebuild_packmol: missing $PACKMOL_SRC/CMakeLists.txt" >&2
  exit 1
fi

if [[ "$FORCE" != 1 && "$CLEAN" != 1 ]] && _binary_runs_on_host "$PACKMOL_OUT"; then
  echo "Reusing $PACKMOL_OUT ($(file -b "$PACKMOL_OUT" 2>/dev/null || echo native))"
  exit 0
fi

if [[ "$CLEAN" == 1 ]]; then
  echo "Cleaning $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

FC="${FC:-${PACKMOL_FC:-$(command -v gfortran || true)}}"
if [[ -z "$FC" ]]; then
  echo "rebuild_packmol: gfortran not found (install gcc/gfortran or set FC=...)" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
echo "Configuring Packmol in $BUILD_DIR (FC=$FC) ..."
cmake -S "$PACKMOL_SRC" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_Fortran_FLAGS_RELEASE:STRING="-O2 -DNDEBUG"

echo "Building packmol ..."
cmake --build "$BUILD_DIR" -j "$(_build_jobs)"
cmake --install "$BUILD_DIR"

BUILT=""
for candidate in \
  "$BUILD_DIR/install/bin/packmol" \
  "$BUILD_DIR/packmol" \
  "$BUILD_DIR/bin/packmol"; do
  if [[ -f "$candidate" ]]; then
    BUILT="$candidate"
    break
  fi
done
if [[ -z "$BUILT" ]]; then
  BUILT="$(find "$BUILD_DIR" -name packmol -type f -perm -111 -print -quit || true)"
fi
if [[ -z "$BUILT" ]]; then
  echo "rebuild_packmol: build finished but packmol executable not found" >&2
  exit 1
fi

cp -f "$BUILT" "$PACKMOL_OUT"
chmod +x "$PACKMOL_OUT"
echo "Installed $PACKMOL_OUT (from $BUILT)"
echo "Verify: $PACKMOL_OUT --help | head -3"
