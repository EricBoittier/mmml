#!/bin/bash
#SBATCH --job-name=bonded-scan
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/collapse_decompose/bonded-scan-%A_%a.out
#SBATCH --error=artifacts/collapse_decompose/bonded-scan-%A_%a.err
#
# Measure --ml-potential-mode bonded_intra on the O-H scan, replacing the
# arithmetic in docs/hybrid-bonded-intra.md with a real curve.
#
# Runs from a synced copy of the feature branch ($HOME/mmml_bonded) on PYTHONPATH,
# borrowing $HOME/mmml_npt's venv and artifacts -- mmml_npt is on main and has no
# bonded_intra. PYTHONPATH must win over the editable install or the old tree is
# silently imported instead.
#
#   0  bonded_only   bonded intra, ML dimer OFF
#        SELF-CHECK. Must reproduce the standalone CGenFF curve from
#        scripts/bonded_vs_ml_intramolecular.py: min at 0.950 A, +342.3 kcal/mol
#        at 1.831 A, monotone below the minimum. If it does not, the wiring is
#        wrong and arm 1 means nothing.
#
#   1  bonded_full   bonded intra + ML dimer interaction
#        THE PROPOSED POTENTIAL. Predicted from separately-measured arms:
#        +347.2 kcal/mol at 1.831 A, and a -58.2 kcal/mol well at 0.770 A that
#        this mode does NOT fix. Both predictions are under test.
set -uo pipefail

SRC="${MMML_SRC:-$HOME/mmml_bonded}"
REPO="${MMML_REPO:-$HOME/mmml_npt}"
export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
# shellcheck disable=SC1091
source "$REPO/scripts/scicore_env.sh"
if [[ ! -f "$REPO/setup/charmm/libcharmm.so" && -f "$HOME/mmml/setup/charmm/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$HOME/mmml/setup/charmm" CHARMM_HOME="$HOME/mmml/setup/charmm"
fi
source "$REPO/.venv/bin/activate"
# cd into the SOURCE tree, not the artifact repo: python puts the cwd at
# sys.path[0], ahead of PYTHONPATH, so running from $REPO silently imports the
# old tree no matter what PYTHONPATH says. Every artifact path below is absolute.
cd "$SRC" || exit 1
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC=1 UV_OFFLINE=1

OUT_ROOT="$REPO/artifacts/collapse_decompose"
CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
SCALES="${SCALES:-$REPO/artifacts/npt_argon_water/scales/scales_trained.json}"
PSF="${PSF:-$REPO/artifacts/npt_argon_water/boxes/tip3_298k/model.psf}"
NPZ="$OUT_ROOT/intramolecular_scan.npz"

[[ -f "$NPZ" ]] || { echo "missing $NPZ" >&2; exit 2; }
[[ -f "$PSF" ]] || { echo "missing $PSF" >&2; exit 2; }

# Prove the branch under test is the one being imported. Grep the printed path
# rather than trusting the exit status: importing mmml pulls in pycharmm, whose
# Fortran STOP overwrites the status with 0 at teardown, so a `|| exit` guard
# here silently passes even when the assert fires. That is how the first run of
# this job reached srun against the old tree.
python -c "
import mmml.cli.run.md_system as ms
print('MMML_IMPORTED_FROM', ms.__file__)
print('HAS_BONDED_INTRA', any(a.dest == 'ml_potential_mode' for a in ms.build_parser()._actions))
" 2>&1 | tee "$OUT_ROOT/import_check_${SLURM_ARRAY_TASK_ID:-0}.txt"
grep -q "HAS_BONDED_INTRA True" "$OUT_ROOT/import_check_${SLURM_ARRAY_TASK_ID:-0}.txt" || {
  echo "ABORT: imported tree has no --ml-potential-mode"; grep MMML_IMPORTED_FROM "$OUT_ROOT/import_check_${SLURM_ARRAY_TASK_ID:-0}.txt"; exit 3; }

ARMS=("bonded_only" "bonded_full")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
arm="${ARMS[$IDX]}"
out="$OUT_ROOT/scan_$arm"
rm -rf "$out"; mkdir -p "$out"

case "$arm" in
  bonded_only) terms=(--include-mm --do-ml --skip-ml-dimers) ;;
  bonded_full) terms=(--include-mm --do-ml --do-ml-dimer) ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac

n_frames=$(python -c "
import numpy as np, sys
print(int(np.load(sys.argv[1])['R'].shape[0]))" "$NPZ")

echo "=== [$IDX] arm=$arm frames=$n_frames"
echo "    terms: ${terms[*]}"
echo "    psf:   $PSF"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

scale_flags=()
[[ -f "$SCALES" ]] && scale_flags=(--mm-lj-scales-file "$SCALES")

python -m mmml.cli md-system \
  --backend jaxmd --setup pbc_nve \
  --composition "TIP3:2" \
  --box-size 30.0 \
  --checkpoint "$CKPT/epoch-25" \
  --ml-potential-mode bonded_intra \
  --jax-mm-spoof-psf "$PSF" \
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
# Never trust the status alone: CHARMM's Fortran STOP overwrites it at teardown.
if [[ -f "$out/evaluate.npz" ]]; then
  python -c "
import numpy as np, sys
E = np.asarray(np.load(sys.argv[1])['E_eV']).reshape(-1)   # E is hartree; E_eV is eV
print(f'  frames {E.size}  span {(E.max()-E.min())*23.060548:.2f} kcal/mol')
" "$out/evaluate.npz"
else
  echo "  NO evaluate.npz -- job failed regardless of the exit status above"
  tail -12 "$out/eval.log" | cut -c1-120
fi
