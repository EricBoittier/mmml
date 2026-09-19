#!/usr/bin/env bash
# Zero-field efield-train with polarizability loss (dμ/dEf at Ef=0).
#
# Usage:
#   scripts/spice_alpha/train_efield_polar.sh splits_des_mono [ckpt_dir] [epochs]

set -euo pipefail

# Do not put a closing brace inside `${1:?...}` (bash ends the expansion
# there). A train/valid brace list made SPLITS `.../splits_des_mono.npz}`.
SPLITS="${1:?splits directory with train and valid NPZs}"
CKPT="${2:-./ckpts/spice_des_mono_efield_polar}"
EPOCHS="${3:-100}"
if [[ ! -d "$SPLITS" ]]; then
  echo "SPLITS is not a directory: $SPLITS" >&2
  exit 1
fi

# SciCORE prolog defaults JAX_ENABLE_X64=1. e3x Embed stays float32 while
# MessagePass promotes, and EFieldPhysNet.init raises in e3x.nn.add.
# load_ef_npz is float32; keep the process on float32 for efield-train.
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
# Polar jacrev autotune OOMs (login12: "All configs failed during profiling"
# on MessagePass transpose). Level 0 skips that profiler.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

BATCH_SIZE="${BATCH_SIZE:-64}"
FEATURES="${FEATURES:-32}"
MAX_DEGREE="${MAX_DEGREE:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
NUM_BASIS_FUNCTIONS="${NUM_BASIS_FUNCTIONS:-10}"
CUTOFF="${CUTOFF:-10.0}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
POLAR_WEIGHT="${POLAR_WEIGHT:-1.0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/check_efield_npz.py" ]]; then
  python "$HERE/check_efield_npz.py" \
    "$SPLITS/energies_forces_dipoles_train.npz" \
    "$SPLITS/energies_forces_dipoles_valid.npz"
fi

EXTRA=()
if [[ "${GRADIENT_CHECKPOINT:-0}" != "0" ]]; then
  EXTRA+=(--gradient-checkpoint)
fi

mmml efield-train \
  --train-npz "$SPLITS/energies_forces_dipoles_train.npz" \
  --valid-npz "$SPLITS/energies_forces_dipoles_valid.npz" \
  --output-dir "$CKPT" \
  --energy_weight "$ENERGY_WEIGHT" \
  --forces_weight 100.0 \
  --dipole_weight 0.1 \
  --polar_weight "$POLAR_WEIGHT" \
  --polar-at-zero-field \
  --field_scale 0.001 \
  --num_epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --features "$FEATURES" \
  --max_degree "$MAX_DEGREE" \
  --num_iterations "$NUM_ITERATIONS" \
  --num_basis_functions "$NUM_BASIS_FUNCTIONS" \
  --cutoff "$CUTOFF" \
  "${EXTRA[@]}"
