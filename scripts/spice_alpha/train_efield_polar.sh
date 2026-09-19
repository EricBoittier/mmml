#!/usr/bin/env bash
# Zero-field efield-train with polarizability loss (dμ/dEf at Ef=0).
#
# Usage:
#   scripts/spice_alpha/train_efield_polar.sh splits_des_mono [ckpt_dir] [epochs]

set -euo pipefail

SPLITS="${1:?directory with energies_forces_dipoles_{train,valid}.npz}"
CKPT="${2:-./ckpts/spice_des_mono_efield_polar}"
EPOCHS="${3:-100}"

mmml efield-train \
  --train-npz "$SPLITS/energies_forces_dipoles_train.npz" \
  --valid-npz "$SPLITS/energies_forces_dipoles_valid.npz" \
  --output-dir "$CKPT" \
  --energy_weight 1.0 \
  --forces_weight 100.0 \
  --dipole_weight 0.1 \
  --polar_weight 1.0 \
  --polar-at-zero-field \
  --field_scale 0.001 \
  --num_epochs "$EPOCHS" \
  --batch_size 64 \
  --features 32 \
  --max_degree 2 \
  --num_iterations 2 \
  --cutoff 10.0
