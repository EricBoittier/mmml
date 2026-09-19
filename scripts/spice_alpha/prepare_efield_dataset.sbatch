#!/usr/bin/env bash
# CPU convert of DES370K monomers → efield NPZ (all frames, skip 1.3 GB dimers).
# Writes a *new* tree so the 256-frame smoke splits stay put.
#
#   sbatch scripts/spice_alpha/prepare_efield_dataset.sbatch
#
# Do not source scripts/scicore_env.sh (not needed; convert is numpy/h5py).
#
#SBATCH --job-name=spice-ef-prep
#SBATCH --partition=scicore
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=artifacts/spice_ef_polar/slurm-prep-%j.out
#SBATCH --error=artifacts/spice_ef_polar/slurm-prep-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
mkdir -p artifacts/spice_ef_polar

if [[ -z "${MMML_PYTHON:-}" && ! -x "$ROOT/.venv/bin/python" && -x "$HOME/mmml/.venv/bin/python" ]]; then
  export MMML_PYTHON="$HOME/mmml/.venv/bin/python"
fi
if [[ -f "$ROOT/scripts/resolve_mmml_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/resolve_mmml_env.sh"
  mmml_resolve_env "$ROOT"
  export PATH="$(dirname "${MMML_PYTHON}"):$PATH"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

SPICE="${SPICE:-$HOME/data/spicealpha}"
OUT="${OUT:-$HOME/data/spicealpha/mmml_efield_full}"
MAX_FRAMES="${MAX_FRAMES:-0}"

export SKIP_DIMERS="${SKIP_DIMERS:-1}"
"$ROOT/scripts/spice_alpha/prepare_efield_dataset.sh" "$SPICE" "$OUT" "$MAX_FRAMES"

python "$ROOT/scripts/spice_alpha/check_efield_npz.py" \
  "$OUT/splits_des_mono/energies_forces_dipoles_train.npz" \
  "$OUT/splits_des_mono/energies_forces_dipoles_valid.npz"

echo "PASS: full monomer NPZ at $OUT/splits_des_mono"
echo "train: SPLITS=$OUT/splits_des_mono CKPT=$ROOT/ckpts/spice_ef_polar_full MODE=full"
