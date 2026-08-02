#!/bin/bash
#SBATCH --job-name=collapse-decomp
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/collapse_decompose/%x-%A_%a.out
#SBATCH --error=artifacts/collapse_decompose/%x-%A_%a.err
#
# Why the bulk NVT run releases energy without bound.
#
# The descent is NOT an intermolecular collapse: the closest O-O contact holds
# at 2.27 -> 2.36 A across the whole trajectory. The molecules come apart
# internally -- O-H bonds spread 0.95-0.99 -> 0.57-1.83 A while E_pot falls to
# -350 kcal/mol per water by the last recorded frame, still falling, with the
# thermostat pulling heat out (T peaks at 1448 K then drops to 830 K).
#
# The DES training dimers are perfectly rigid: O-H = 0.9840 A, std exactly 0.0,
# HOH = 104.60 deg, in all 295 of them. setup_calculator gives the ML term
# ownership of the internal monomer energy, so the model owns a surface it has
# exactly one training point for.
#
# Two datasets:
#
#   scan    one O-H of a dimer scanned 0.55 -> 1.95 A, everything else frozen at
#           the training geometry, at the liquid first-peak O-O of 2.754 A.
#           MECHANISM: a sound potential has a steep minimum near 0.984 A; one
#           fit to rigid monomers has no wall and can fall away without bound.
#
#   frames  the recorded frames of the failing NVT trajectory.
#           ATTRIBUTION: differencing the arms says which term released it.
#
# Arms (identical geometries in every arm, so they difference exactly):
#
#   full      ML + ML-dimer + MM    the potential the campaign actually ran
#   no_dimer  ML + MM               removes the learned dimer correction
#   no_mm     ML + ML-dimer         removes CGenFF LJ + electrostatics
#   mm_only   MM                    the classical baseline alone
#
#   full - no_mm  = the MM term        full - no_dimer = the ML dimer term
#
# Single points only, so this is minutes. Deliberately run from the SAME tree
# that produced the collapsing trajectory ($HOME/mmml_npt), so the potential
# being decomposed is the one that actually failed, not a rebuilt approximation.
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

OUT_ROOT="$REPO/artifacts/collapse_decompose"
CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
SCALES="${SCALES:-$REPO/artifacts/npt_argon_water/scales/scales_trained.json}"
mkdir -p "$OUT_ROOT"

ARMS=("full" "no_dimer" "no_mm" "mm_only")
DATASETS=("scan" "frames")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
arm="${ARMS[$((IDX % 4))]}"
dset="${DATASETS[$((IDX / 4))]}"

case "$dset" in
  scan)
    NPZ="$OUT_ROOT/intramolecular_scan.npz"
    COMP="TIP3:2"
    # Far larger than any real pair separation here, so minimum image never
    # wraps one; a box is still required or the MM pair list finds nothing.
    BOXSIZE="30.0"
    ;;
  frames)
    NPZ="$OUT_ROOT/collapse_frames.npz"
    COMP="TIP3:732"
    BOXSIZE="28.0"
    ;;
  *) echo "bad dataset $dset" >&2; exit 2 ;;
esac

[[ -f "$NPZ" ]] || {
  echo "missing $NPZ -- run scripts/make_collapse_diagnostics_npz.py first" >&2
  exit 2
}

out="$OUT_ROOT/${dset}_${arm}"
rm -rf "$out"; mkdir -p "$out"

case "$arm" in
  full)     terms=(--include-mm --do-ml --do-ml-dimer) ;;
  no_dimer) terms=(--include-mm --do-ml --skip-ml-dimers) ;;
  no_mm)    terms=(--no-include-mm --do-ml --do-ml-dimer) ;;
  mm_only)  terms=(--include-mm --no-do-ml --skip-ml-dimers) ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac

n_frames=$(uv run python -c "
import numpy as np, sys
print(int(np.load(sys.argv[1])['R'].shape[0]))" "$NPZ")

echo "=== [$IDX] dataset=$dset arm=$arm frames=$n_frames comp=$COMP box=$BOXSIZE"
echo "    terms: ${terms[*]}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

# --mm-lj-scales-file is only meaningful when the MM term is present.
scale_flags=()
if [[ "$arm" != "no_mm" && -f "$SCALES" ]]; then
  scale_flags=(--mm-lj-scales-file "$SCALES")
fi

uv run mmml md-system \
  --backend jaxmd --setup pbc_nve \
  --composition "$COMP" \
  --box-size "$BOXSIZE" \
  --checkpoint "$CKPT/epoch-25" \
  "${scale_flags[@]}" "${terms[@]}" \
  --evaluate-npz "$NPZ" \
  --evaluate-reference-npz "$NPZ" \
  --max-frames "$n_frames" \
  --evaluate-output "$out/evaluate.json" \
  --evaluate-forces-npz "$out/evaluate.npz" \
  --output-dir "$out" \
  --ml-batch-size "${ML_BATCH:-256}" \
  --quiet > "$out/eval.log" 2>&1
rc=$?

echo "--- exit $rc"
if [[ -f "$out/evaluate.npz" ]]; then
  # Read E_eV, not E. E is stored as eV * EV_TO_HARTREE; reading it as eV is
  # exactly the 27.211x error that produced the bogus r = 0.0008 diagnostic.
  uv run python -c "
import numpy as np, sys
d = np.load(sys.argv[1])
E = np.asarray(d['E_eV']).reshape(-1)
print(f'  frames evaluated {E.size}')
print(f'  E range {E.min():.4f} .. {E.max():.4f} eV  (span {(E.max()-E.min())*23.060548:.2f} kcal/mol)')
" "$out/evaluate.npz"
else
  echo "  no evaluate.npz written"
  tail -8 "$out/eval.log" | cut -c1-120
fi
exit $rc
