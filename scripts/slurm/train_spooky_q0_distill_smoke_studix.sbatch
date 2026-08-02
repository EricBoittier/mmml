#!/usr/bin/env bash
# Experiment 2 (distilled charge student) -- SMOKE.
#
# Must complete and write a real epoch checkpoint carrying teacher provenance
# before the full distilled job is submitted. Mirrors the plain q0 retrain
# smoke so the only difference under test is the teacher.
#
# TEACHER CHOICE. Not step-00294400: that run was trained on the DES cache WITH
# the CGenFF LJ term and mol_id, neither of which exists in the student's SO3LR
# cache. Evaluated here its LJ prior, its learned per-atom vdW-scale head and its
# element/global LJ scales are all inert and its electrostatics runs unmasked, so
# its outputs would not be what it was trained to produce. The trainer's
# architecture gate refuses it outright (8 checkpoint parameters with nowhere to
# go) rather than distilling from a silently half-loaded model.
#
# spooky_so3lr_adam_cw2/step-02700000 is compatible: AdamW (not Muon), trained on
# a CGenFF-free / mol_id-free SO3LR cache, same architecture family
# (features 128, max_degree 2, efa off), trainable_zbl explicitly recorded. It is
# also the only adam_cw2 checkpoint with a completed evaluation and a TIP3 dimer
# scan -- the surface that actually fails.
#
# Its architecture still differs from the student (128 vs 16 features, degree 2
# vs 1, efa off vs on): a real capacity gap, which is the normal distillation
# direction. The teacher's charge head is deliberately NOT distilled either way.
#
# STILL OUTSTANDING: this teacher has no max-force measurement on the prepared
# 732-water box, the way step-00294400's 3.50 eV/A was obtained. Gate it before
# trusting a full run.
#
# The run is bounded with --steps-per-epoch, NOT --max-structures: that flag is
# only read while building a cache (cache_extxyz_to_orbax), so under
# `--mode train --cache-path` it silently does nothing and the "smoke" becomes a
# full ~924k-step epoch that the wall clock kills before any checkpoint is
# written. --steps-per-epoch ends the epoch for real, so validation runs and
# --save-every 1 produces the checkpoint this smoke exists to produce.
#SBATCH --job-name=q0-distill-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/spooky_q0_distill_smoke/slurm-%j.out
#SBATCH --error=artifacts/spooky_q0_distill_smoke/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
# A side checkout (e.g. ~/mmml_distill) has no .venv of its own. Borrow the main
# checkout's interpreter, but put THIS tree first on PYTHONPATH: that venv holds
# an editable install pointing at ~/mmml, so without this `import mmml` silently
# loads the other tree's code and the run tests something you did not change.
if [[ -z "${MMML_PYTHON:-}" && ! -x "$ROOT/.venv/bin/python" && -x "$HOME/mmml/.venv/bin/python" ]]; then
  export MMML_PYTHON="$HOME/mmml/.venv/bin/python"
fi
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Verify rather than assume: refuse to run if `import mmml` resolves elsewhere.
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
OUT="$ROOT/artifacts/spooky_q0_distill_smoke"
mkdir -p "$OUT"

for required in "$CACHE/_METADATA" "$INIT/_CHECKPOINT_METADATA" "$TEACHER/_CHECKPOINT_METADATA"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 66; }
done

"$MMML_PYTHON" scripts/train_so3lr_spooky_extxyz.py \
  --mode train --cache-path "$CACHE" \
  --workdir "$OUT" --init-checkpoint "$INIT" \
  --teacher-checkpoint "$TEACHER" \
  --distill-alpha 0.75 --distill-targets energy forces \
  --distill-energy-align atomic --distill-align-batches 16 \
  --epochs 1 --steps-per-epoch 40 --num-devices 1 \
  --batch-size-per-device 4 --max-pairs-per-device 18000 \
  --atom-bucket-width 4 --auto-batch --auto-batch-margin 1 \
  --optimizer adamw --learning-rate 2.5e-5 \
  --lr-schedule warmup_cosine --lr-warmup-steps 50 --lr-decay-steps 0 \
  --lr-end-fraction 0.05 --weight-decay 0.0 --clip-global-norm 1.0 \
  --energy-weight 1.0 --forces-weight 75.0 \
  --dipole-weight 5.0 --charges-weight 5.0 \
  --features 16 --max-degree 1 --num-iterations 3 \
  --num-basis-functions 32 --cutoff 6.0 --max-atomic-number 87 --n-res 2 \
  --predict-charges --trainable-zbl --zbl-cuton 0.1 --zbl-cutoff 0.6 \
  --electrostatics-damping-sigma 4.0 \
  --valid-fraction 0.05 --valid-steps 10 \
  --save-every 1 --seed 0 --log-every 10 --log-every-steps 10

# The point of the smoke: a checkpoint that records WHICH teacher it used.
[[ -s "$OUT/distillation.json" ]] || {
  echo "smoke produced no distillation.json -- teacher provenance was not recorded" >&2
  exit 67
}
echo "distillation provenance:"
cat "$OUT/distillation.json"
