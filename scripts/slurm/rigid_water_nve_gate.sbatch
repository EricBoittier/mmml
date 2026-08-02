#!/bin/bash
#SBATCH --job-name=rigid-nve
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/rigid_water/rigid-nve-%A_%a.out
#SBATCH --error=artifacts/rigid_water/rigid-nve-%A_%a.err
#
# Does the DES hybrid give a sensible liquid once the monomers are held rigid?
#
# Unconstrained, bulk TIP3 under this checkpoint releases 350.5 kcal/mol per
# water and is still falling: the ML term was trained on perfectly rigid DES
# dimers (O-H = 0.9840 A, std exactly 0.0) and so carries no intramolecular
# restoring force. Constraining the monomers makes MD sample the distribution
# the model was actually trained on.
#
# NVE, because it is the sharpest gate available: with no thermostat, E_tot must
# be conserved. Drift means the forces are not the gradient of the energy, which
# no ensemble choice can hide. No density number should be believed from a
# potential that fails this.
#
# Drift rescue and the drift abort are BOTH disabled. They exist to keep a
# production run alive, but here they destroy the measurement: the rescue path
# rewinds, rethermalizes and runs CHARMM SD/ABNR, which moves atoms outside the
# constraint projection and reseeds velocities. A gate that repairs the thing it
# is measuring reports nothing. The first attempt at this run hit rescue
# attempt 4/5 and had to be thrown away.
#
#   0  rigid    --rigid-water     the proposed configuration
#   1  flexible (control)         reproduces the collapse; without it arm 0
#                                 proves nothing
#
# Runs from the synced feature tree; cd into it because python puts the cwd at
# sys.path[0] ahead of PYTHONPATH.
set -uo pipefail

SRC="${MMML_SRC:-$HOME/mmml_rigid}"
REPO="${MMML_REPO:-$HOME/mmml_npt}"
export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
# shellcheck disable=SC1091
source "$REPO/scripts/scicore_env.sh"
if [[ ! -f "$REPO/setup/charmm/libcharmm.so" && -f "$HOME/mmml/setup/charmm/libcharmm.so" ]]; then
  export CHARMM_LIB_DIR="$HOME/mmml/setup/charmm" CHARMM_HOME="$HOME/mmml/setup/charmm"
fi
source "$REPO/.venv/bin/activate"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC=1 UV_OFFLINE=1
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64

OUT_ROOT="$REPO/artifacts/rigid_water"
mkdir -p "$OUT_ROOT"
CKPT="${CKPT:-$REPO/artifacts/lj_scales_des_production/seed_42/ckpts/hybrid_mm_scaled_lj_des_full_seed42-5c75054c-bab3-4b9f-99bd-79d5b7fd7307}"
BOX="$REPO/artifacts/npt_argon_water/boxes/tip3_298k"
SCALES="$REPO/artifacts/npt_argon_water/scales/scales_unit.json"

# Prove the tree under test is the one imported. Grep a printed marker rather
# than trusting an exit status: importing mmml pulls in pycharmm, whose Fortran
# STOP overwrites the status with 0 at teardown.
python -c "
import mmml.cli.run.jaxmd_runner as r
from mmml.md.constraints import maybe_wrap_rigid_water
print('MMML_FROM', r.__file__)
print('HAS_RIGID_WATER True')
" 2>&1 | tee "$OUT_ROOT/import_check_${SLURM_ARRAY_TASK_ID:-0}.txt"
grep -q "HAS_RIGID_WATER True" "$OUT_ROOT/import_check_${SLURM_ARRAY_TASK_ID:-0}.txt" || {
  echo "ABORT: imported tree has no rigid-water support"; exit 3; }

ARMS=("rigid" "flexible")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
arm="${ARMS[$IDX]}"
out="$OUT_ROOT/$arm"
rm -rf "$out"; mkdir -p "$out"

rigid_flags=()
[[ "$arm" == "rigid" ]] && rigid_flags=(--rigid-water)

read -r nmol side < <(python -c "
import json,sys
b=json.load(open(sys.argv[1]))
print(int(b['n_molecules']), float(b['box_side_A']))" "$BOX/box.json") || exit 5

echo "=== [$IDX] arm=$arm  ${nmol} TIP3, ${side} A, ${PS:-2} ps"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

python -m mmml.cli md-system \
  --backend jaxmd --setup pbc_nve \
  --composition "TIP3:${nmol}" \
  --from-psf "$BOX/model.psf" --from-crd "$BOX/model.crd" \
  --skip-cluster-build --box-size "$side" \
  --checkpoint "$CKPT/epoch-25" \
  --mm-lj-scales-file "$SCALES" \
  --temperature 298.15 \
  --ps "${PS:-2}" --dt-fs "${DT:-0.25}" \
  --output-dir "$out" \
  --include-mm --do-ml --do-ml-dimer \
  --ml-batch-size "${ML_BATCH:-1024}" \
  --seed 123 \
  --no-calculator-pre-minimize --no-charmm-pre-minimize \
  --fire-min-steps 0 \
  --jaxmd-minimize-steps "${MINSTEPS:-50}" --jaxmd-pbc-minimize-steps "${MINSTEPS:-50}" \
  --no-handoff-quality-gate \
  --mm-charge-mode fixed \
  --jax-md-skin-distance "${SKIN:-0.5}" --jax-md-update-interval "${NLINT:-1}" \
  --steps-per-recording "${REC:-20}" \
  --no-nve-etot-drift-rescue \
  --nve-etot-drift-abort-eV 0 \
  "${rigid_flags[@]}" > "$out/md.log" 2>&1
rc=$?

echo "--- exit $rc"
ls "$out"/*.traj >/dev/null 2>&1 || { echo "  NO trajectory written -- failed regardless of the status above"; tail -15 "$out/md.log" | cut -c1-120; exit 1; }
python "$SRC/scripts/analyze_rigid_water_run.py" \
  --traj "$(ls "$out"/*.traj | head -1)" \
  --n-monomers "$nmol" \
  --output "$out/analysis.json" 2>&1 | tail -40
