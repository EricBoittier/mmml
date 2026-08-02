#!/usr/bin/env bash
#SBATCH --job-name=q0-longer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
#SBATCH --time=2-00:00:00
#SBATCH --output=artifacts/spooky_q0_longer/slurm-%j.out
#SBATCH --error=artifacts/spooky_q0_longer/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CACHE="/mmhome/boittier/home/so3lr_orbax_cache/so3lr_train_flat_5ba739ddac63717d"
INIT="$ROOT/artifacts/spooky_so3lr_charges/epoch-0002"
OUT="$ROOT/artifacts/spooky_q0_longer"
mkdir -p "$OUT"

for required in "$CACHE/_METADATA" "$INIT/_CHECKPOINT_METADATA"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 66; }
done

"$MMML_PYTHON" scripts/train_so3lr_spooky_extxyz.py \
  --mode train --cache-path "$CACHE" \
  --workdir "$OUT" --init-checkpoint "$INIT" \
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
