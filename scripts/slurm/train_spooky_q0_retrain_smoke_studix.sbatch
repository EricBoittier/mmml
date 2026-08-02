#!/usr/bin/env bash
#SBATCH --job-name=q0-retrain-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/spooky_q0_retrain_smoke/slurm-%j.out
#SBATCH --error=artifacts/spooky_q0_retrain_smoke/slurm-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$MMML_PYTHON" scripts/train_so3lr_spooky_extxyz.py \
  --mode train \
  --cache-path /mmhome/boittier/home/so3lr_orbax_cache/so3lr_train_flat_5ba739ddac63717d \
  --workdir "$ROOT/artifacts/spooky_q0_retrain_smoke" \
  --init-checkpoint "$ROOT/artifacts/spooky_so3lr_charges/epoch-0002" \
  --max-structures 512 --epochs 1 --num-devices 1 \
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
