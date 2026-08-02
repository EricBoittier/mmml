#!/bin/bash
#SBATCH --job-name=bonded-intra
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/collapse_decompose/bonded-intra-%j.out
#SBATCH --error=artifacts/collapse_decompose/bonded-intra-%j.err
#
# The premise test for "let a bonded model own the intramolecular coordinate".
#
# The hybrid builds its dimer term as E_AB - (E_A + E_B) from one ML model. Along
# an O-H scan the monomer sum rises (+46.5 kcal/mol at 0.79 A, correct in sign)
# but the dimer total falls (-18.8), so the interaction inherits a -73 kcal/mol
# well at 0.77 A that is cancellation error, not physics.
#
# This measures what the CGenFF bonded model does on the identical scan. If it
# has a proper minimum near b0 and rises monotonically below it, then intra is a
# solved problem in the MM term and the ML only ever needs to carry the
# interaction -- which is weakly dependent on monomer internals, so the rigid DES
# training set is sufficient for it.
set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml_npt}"
cd "$REPO" || exit 1
export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
# shellcheck disable=SC1091
source scripts/scicore_env.sh
if [[ ! -f "$REPO/setup/charmm/libcharmm.so" && -f "$HOME/mmml/setup/charmm/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$HOME/mmml/setup/charmm" CHARMM_HOME="$HOME/mmml/setup/charmm"
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC=1 UV_OFFLINE=1

OUT="$REPO/artifacts/collapse_decompose"

uv run python scripts/bonded_vs_ml_intramolecular.py \
  --scan-npz "$OUT/intramolecular_scan.npz" \
  --psf "$REPO/artifacts/npt_argon_water/boxes/tip3_298k/model.psf" \
  --output "$OUT/bonded_intramolecular.npz"
echo "--- exit $?"
