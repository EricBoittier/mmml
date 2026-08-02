#!/bin/bash
#SBATCH --job-name=dimer2d
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/dimer_2d/%x-%A_%a.out
#SBATCH --error=artifacts/dimer_2d/%x-%A_%a.err
#
# Evaluate the same 2D water-dimer grid with terms switched on and off. Two
# questions, one experiment:
#
#   1. WHICH TERM supplies the -99.3 kcal/mol per molecule that the hybrid
#      releases in bulk water. Differencing the arms below isolates it.
#   2. IS THE DIMER SURFACE SANE -- the gate on any further condensed-phase
#      work. A spurious attractive well is visible directly on the surface.
#
# Arms (all on the identical 600-frame grid, so they difference exactly):
#
#   0  full      ML + ML-dimer + MM     the potential the campaign actually ran
#   1  no_dimer  ML + MM                removes the learned dimer correction
#   2  no_mm     ML + ML-dimer          removes CGenFF LJ + electrostatics
#   3  mm_only   MM                     the classical baseline on its own
#
#   full - no_dimer  = the ML dimer term
#   full - no_mm     = the MM term
#
# Single-point only (--evaluate-npz), no dynamics, so this is minutes not hours.
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

OUT_ROOT="$REPO/artifacts/dimer_2d"
GRID="${GRID:-$OUT_ROOT/water_dimer_2d.npz}"
CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
SCALES="${SCALES:-$REPO/artifacts/npt_argon_water/scales/scales_trained.json}"
mkdir -p "$OUT_ROOT"

[[ -f "$GRID" ]] || { echo "missing grid $GRID (run scripts/make_dimer_2d_grid.py)" >&2; exit 2; }

ARMS=("full" "no_dimer" "no_mm" "mm_only")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
arm="${ARMS[$IDX]}"
out="$OUT_ROOT/$arm"
rm -rf "$out"; mkdir -p "$out"

case "$arm" in
  full)     terms=(--include-mm --do-ml --do-ml-dimer) ;;
  no_dimer) terms=(--include-mm --do-ml --skip-ml-dimers) ;;
  no_mm)    terms=(--no-include-mm --do-ml --do-ml-dimer) ;;
  mm_only)  terms=(--include-mm --no-do-ml --skip-ml-dimers) ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac

n_frames=$(uv run python -c "
import numpy as np,sys
print(int(np.load(sys.argv[1])['R'].shape[0]))" "$GRID")

echo "=== [$IDX] arm=$arm  frames=$n_frames"
echo "    terms: ${terms[*]}"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

# --mm-lj-scales-file is only meaningful when the MM term is present.
scale_flags=()
if [[ "$arm" != "no_mm" && -f "$SCALES" ]]; then
  scale_flags=(--mm-lj-scales-file "$SCALES")
fi

uv run mmml md-system \
  --backend jaxmd --setup pbc_nve \
  --composition "TIP3:2" \
  --checkpoint "$CKPT/epoch-25" \
  "${scale_flags[@]}" "${terms[@]}" \
  --evaluate-npz "$GRID" \
  --evaluate-reference-npz "$GRID" \
  --max-frames "$n_frames" \
  --evaluate-output "$out/evaluate.json" \
  --evaluate-forces-npz "$out/evaluate.npz" \
  --output-dir "$out" \
  --ml-batch-size "${ML_BATCH:-256}" \
  --quiet > "$out/eval.log" 2>&1
rc=$?

echo "--- exit $rc"
if [[ -f "$out/evaluate.npz" ]]; then
  uv run python -c "
import numpy as np,sys
d=np.load(sys.argv[1])
E=np.asarray(d['E']).reshape(-1)
print(f'  frames evaluated {E.size}')
print(f'  E range {E.min():.4f} .. {E.max():.4f} eV')
" "$out/evaluate.npz"
else
  echo "  no evaluate.npz written"
  tail -6 "$out/eval.log" | cut -c1-110
fi
exit $rc
