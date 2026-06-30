#!/usr/bin/env bash
# Build shared (-fPIC) FFTW 3.3.10 for linking into libcharmm.so.
#
# Cluster /opt FFTW installs are often static-only; linking .a into a shared
# library fails with "recompile with -fPIC". Run this once per machine, then:
#
#   export MMML_FFTW_ROOT=$HOME/.local/fftw-3.3.10-pic
#   OPENMPI_ROOT=/opt/gcc-12.2.0/openmpi-4.1.4/build bash scripts/rebuild_charmm_mlpot.sh
#
# Usage:
#   bash scripts/build_fftw_pic.sh
#   MMML_FFTW_ROOT=$HOME/fftw-pic bash scripts/build_fftw_pic.sh

set -euo pipefail

VERSION="${FFTW_VERSION:-3.3.10}"
PREFIX="${MMML_FFTW_ROOT:-${HOME}/.local/fftw-${VERSION}-pic}"
SRC_DIR="${FFTW_SRC_DIR:-${TMPDIR:-/tmp}/fftw-${VERSION}-src}"
TARBALL="${FFTW_TARBALL:-fftw-${VERSION}.tar.gz}"
URL="https://www.fftw.org/${TARBALL}"

echo "FFTW PIC build: prefix=$PREFIX"

if [[ -f "$PREFIX/lib/libfftw3.so" && -f "$PREFIX/lib/libfftw3f.so" ]]; then
  echo "Already built: $PREFIX/lib/libfftw3.so"
  echo "export MMML_FFTW_ROOT=$PREFIX"
  exit 0
fi

mkdir -p "$(dirname "$SRC_DIR")"
if [[ ! -d "$SRC_DIR" ]]; then
  if [[ ! -f "${SRC_DIR%/src}/${TARBALL}" ]]; then
    echo "Downloading $URL ..."
    curl -fsSL "$URL" -o "${SRC_DIR%/src}/${TARBALL}"
  fi
  tar -xzf "${SRC_DIR%/src}/${TARBALL}" -C "$(dirname "$SRC_DIR")"
  mv "$(dirname "$SRC_DIR")/fftw-${VERSION}" "$SRC_DIR"
fi

mkdir -p "$PREFIX"
cd "$SRC_DIR"

# Single prefix: libfftw3 (double) + libfftw3f (single via --enable-float).
./configure \
  --prefix="$PREFIX" \
  --enable-shared \
  --disable-static \
  --enable-float \
  --enable-threads \
  CFLAGS="-fPIC -O2 -march=native"

make -j"$(nproc 2>/dev/null || echo 4)"
make install

echo ""
echo "Done. Add to your rebuild env:"
echo "  export MMML_FFTW_ROOT=$PREFIX"
echo "  export FFTW_ROOT=$PREFIX"
echo "  export FFTWF_ROOT=$PREFIX"
echo "  export LD_LIBRARY_PATH=$PREFIX/lib:\$LD_LIBRARY_PATH"
