#!/usr/bin/env bash
# Build shared (-fPIC) FFTW 3.3.10 for linking into libcharmm.so.
#
# Cluster /opt FFTW installs are often static-only; linking .a into a shared
# library fails with "recompile with -fPIC". Run this once per machine, then:
#
#   export MMML_FFTW_ROOT=$HOME/.local/fftw-3.3.10-pic
#   export FFTW_ROOT=$MMML_FFTW_ROOT FFTWF_ROOT=$MMML_FFTW_ROOT
#   OPENMPI_ROOT=/opt/gcc-12.2.0/openmpi-4.1.4/build bash scripts/rebuild_charmm_mlpot.sh --clean
#
# Usage:
#   bash scripts/build_fftw_pic.sh
#   MMML_FFTW_ROOT=$HOME/fftw-pic bash scripts/build_fftw_pic.sh

set -euo pipefail

VERSION="${FFTW_VERSION:-3.3.10}"
PREFIX="${MMML_FFTW_ROOT:-${HOME}/.local/fftw-${VERSION}-pic}"
BUILD_DIR="${FFTW_BUILD_DIR:-${HOME}/.cache/mmml-fftw-build}"
SRC_DIR="${FFTW_SRC_DIR:-$BUILD_DIR/fftw-${VERSION}}"
TARBALL="${FFTW_TARBALL:-fftw-${VERSION}.tar.gz}"
TARBALL_PATH="$BUILD_DIR/$TARBALL"
URL="https://www.fftw.org/${TARBALL}"

echo "FFTW PIC build: prefix=$PREFIX"
echo "  staging=$BUILD_DIR"

if [[ -f "$PREFIX/lib/libfftw3.so" && -f "$PREFIX/lib/libfftw3f.so" ]]; then
  echo "Already built: $PREFIX/lib/libfftw3.so"
  echo "export MMML_FFTW_ROOT=$PREFIX"
  exit 0
fi

mkdir -p "$BUILD_DIR" "$PREFIX"

_download_tarball() {
  if [[ -f "$TARBALL_PATH" ]]; then
    echo "Using cached tarball: $TARBALL_PATH"
    return 0
  fi
  echo "Downloading $URL ..."
  echo "  → $TARBALL_PATH"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL "$URL" -o "$TARBALL_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$TARBALL_PATH" "$URL"
  else
    echo "build_fftw_pic: need curl or wget to fetch $URL" >&2
    exit 1
  fi
}

if [[ ! -d "$SRC_DIR" ]]; then
  _download_tarball
  echo "Extracting $TARBALL_PATH ..."
  tar -xzf "$TARBALL_PATH" -C "$BUILD_DIR"
  if [[ ! -d "$BUILD_DIR/fftw-${VERSION}" ]]; then
    echo "build_fftw_pic: expected $BUILD_DIR/fftw-${VERSION} after extract" >&2
    exit 1
  fi
  if [[ "$SRC_DIR" != "$BUILD_DIR/fftw-${VERSION}" ]]; then
    rm -rf "$SRC_DIR"
    mv "$BUILD_DIR/fftw-${VERSION}" "$SRC_DIR"
  fi
fi

cd "$SRC_DIR"

# Single prefix: libfftw3 (double) + libfftw3f (single via --enable-float).
if [[ ! -f Makefile ]]; then
  echo "Configuring FFTW in $SRC_DIR ..."
  ./configure \
    --prefix="$PREFIX" \
    --enable-shared \
    --disable-static \
    --enable-float \
    --enable-threads \
    CFLAGS="-fPIC -O2"
fi

make -j"$(nproc 2>/dev/null || echo 4)"
make install

if [[ ! -f "$PREFIX/lib/libfftw3.so" || ! -f "$PREFIX/lib/libfftw3f.so" ]]; then
  echo "build_fftw_pic: install failed — missing .so under $PREFIX/lib" >&2
  ls -la "$PREFIX/lib" 2>/dev/null || true
  exit 1
fi

echo ""
echo "Done. Add to your rebuild env:"
echo "  export MMML_FFTW_ROOT=$PREFIX"
echo "  export FFTW_ROOT=$PREFIX"
echo "  export FFTWF_ROOT=$PREFIX"
echo "  export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH"
