#!/usr/bin/env bash
# Experiment 2 (distilled charge student) -- FULL RUN.
#
# Do not submit until train_spooky_q0_distill_smoke_studix.sbatch has completed
# and written a real epoch checkpoint with teacher provenance.
#
# Matched to train_spooky_q0_longer_studix.sbatch in every respect except the
# teacher, so the three-way comparison isolates distillation:
#   1. longer supervised continuation  -> artifacts/spooky_q0_longer
#   2. teacher-distilled charge student -> artifacts/spooky_q0_distill  (this)
#   3. distilled + failure-cluster enrichment                  (not yet built)
#
# Ground-truth RIMP2 energies/forces stay the primary objective at alpha=0.75;
# the teacher regularizes rather than replaces them. Charges and dipoles are
# supervised by reference data only -- never by the teacher.
#
# Teacher is spooky_so3lr_adam_cw2/step-02700000, NOT step-00294400 -- see the
# smoke script for why (the latter needs CGenFF/mol_id the student's cache lacks).
# That teacher still needs a max-force measurement on the prepared 732-water box
# before this full run is worth its GPU time.
#SBATCH --job-name=q0-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/spooky_q0_distill/slurm-%j.out
#SBATCH --error=artifacts/spooky_q0_distill/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
# See the smoke script: a side checkout must win over the venv's editable
# install, or `import mmml` silently loads the other tree.
if [[ -z "${MMML_PYTHON:-}" && ! -x "$ROOT/.venv/bin/python" && -x "$HOME/mmml/.venv/bin/python" ]]; then
  export MMML_PYTHON="$HOME/mmml/.venv/bin/python"
fi
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$MMML_PYTHON" - "$ROOT" <<'PY'
import sys, pathlib, mmml
root = pathlib.Path(sys.argv[1]).resolve()
got = pathlib.Path(mmml.__file__).resolve()
if root not in got.parents:
    sys.exit(f"import mmml resolved to {got}, not under {root} -- refusing to run")
from mmml.models.physnetjax.physnetjax.training import spooky_distill
print(f"import check OK: mmml={got}, spooky_distill={spooky_distill.__file__}")
PY

CACHE="/mmhome/boittier/home/so3lr_orbax_cache/so3lr_train_flat_5ba739ddac63717d"
INIT="$ROOT/artifacts/spooky_so3lr_charges/epoch-0002"
TEACHER="$ROOT/artifacts/spooky_so3lr_adam_cw2/step-02700000"
OUT="$ROOT/artifacts/spooky_q0_distill"
mkdir -p "$OUT"

for required in "$CACHE/_METADATA" "$INIT/_CHECKPOINT_METADATA" "$TEACHER/_CHECKPOINT_METADATA"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 66; }
done

"$MMML_PYTHON" scripts/train_so3lr_spooky_extxyz.py \
  --mode train --cache-path "$CACHE" \
  --workdir "$OUT" --init-checkpoint "$INIT" \
  --teacher-checkpoint "$TEACHER" \
  --distill-alpha 0.75 --distill-targets energy forces \
  --distill-energy-align atomic --distill-align-batches 32 \
  --epochs 50 --num-devices 2 \
  --batch-size-per-device 4 --max-pairs-per-device 18000 \
  --atom-bucket-width 4 --auto-batch --auto-batch-margin 1 \
  --optimizer adamw --learning-rate 2.5e-5 \
  --lr-schedule warmup_cosine --lr-warmup-steps 500 --lr-decay-steps 0 \
  --lr-end-fraction 0.05 --weight-decay 0.0 --clip-global-norm 1.0 \
  --energy-weight 1.0 --forces-weight 75.0 \
  --dipole-weight 5.0 --charges-weight 5.0 \
  --features 16 --max-degree 1 --num-iterations 3 \
  --num-basis-functions 32 --cutoff 6.0 --max-atomic-number 87 --n-res 2 \
  --predict-charges --trainable-zbl --zbl-cuton 0.1 --zbl-cutoff 0.6 \
  --electrostatics-damping-sigma 4.0 \
  --valid-fraction 0.05 --valid-steps 100 \
  --save-every 1 --save-every-steps 20000 \
  --seed 0 --log-every 10000 --log-every-steps 200
