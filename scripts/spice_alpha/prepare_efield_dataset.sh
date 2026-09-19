#!/usr/bin/env bash
# Convert an unzipped SPICE-α tree (Zenodo 19205036) to efield-train NPZs.
# Does not download anything. Run on the machine that holds the 12 GB zip.
#
# Usage:
#   scripts/spice_alpha/prepare_efield_dataset.sh ~/data/spicealpha [out_dir] [max_frames]
#
# max_frames=0 means all frames. Use 256 for a smoke extract.

set -euo pipefail

ROOT="${1:?unzipped SPICE-α directory (contains SPICE-alpha/SPICE-alpha.tar.gz)}"
OUT="${2:-$ROOT/mmml_efield}"
MAX_FRAMES="${3:-0}"

ROOT="$(cd "$ROOT" && pwd)"
mkdir -p "$OUT"
TAR="$ROOT/SPICE-alpha/SPICE-alpha.tar.gz"
H5_DIR="$ROOT/SPICE-alpha"

if [[ ! -f "$H5_DIR/DES370K_Monomers.hdf5" ]]; then
  if [[ ! -f "$TAR" ]]; then
    echo "missing $TAR — unzip SPICE-alpha.zip first" >&2
    exit 1
  fi
  echo "extracting DES370K monomers/dimers from tarball (members are ./DES370K_*.hdf5)"
  EXTRACT_TAR="$TAR" EXTRACT_DIR="$H5_DIR" python -c \
    'import os; from mmml.data.spice_alpha import extract_des370k_hdf5; extract_des370k_hdf5(os.environ["EXTRACT_TAR"], os.environ["EXTRACT_DIR"])'
fi

MAX_ARGS=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  MAX_ARGS=(--max-frames "$MAX_FRAMES")
fi

echo "converting monomers → efield NPZ (Ef=0, polar in Bohr³, neutrals only)"
python -m mmml.data.spice_alpha \
  "$H5_DIR/DES370K_Monomers.hdf5" \
  -o "$OUT/spice_des_mono.npz" \
  --efield --polar-units bohr3 --neutral-only \
  --split-dir "$OUT/splits_des_mono" \
  --train-frac 0.9 --valid-frac 0.05 --test-frac 0.05 \
  "${MAX_ARGS[@]}"

if [[ -f "$H5_DIR/DES370K_Dimers.hdf5" ]]; then
  echo "converting dimers (optional; pad from data)"
  python -m mmml.data.spice_alpha \
    "$H5_DIR/DES370K_Dimers.hdf5" \
    -o "$OUT/spice_des_dimers.npz" \
    --efield --polar-units bohr3 --neutral-only \
    --split-dir "$OUT/splits_des_dimers" \
    --train-frac 0.9 --valid-frac 0.05 --test-frac 0.05 \
    "${MAX_ARGS[@]}"
fi

echo "wrote $OUT"
echo "train with scripts/spice_alpha/train_efield_polar.sh $OUT/splits_des_mono"
