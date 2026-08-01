#!/bin/bash
# Production scaled-LJ ensemble member. SEED is supplied at submission time.
# The same complete dataset is evaluated in-sample; thermodynamics validate it.
#SBATCH --job-name=des-lj-prod
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_production/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_production/logs/slurm-%j.err

set -euo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
DATASET="$REPO/artifacts/lj_scales_des/des_dimers_cgenff_all.npz"
BASE_RUN="$REPO/artifacts/lj_scales_des_full/ckpts/hybrid_mm_fixed_lj_scales_des_full_insample-155c22fe-5788-42c0-9dc2-fcf04ffdd049"
BASE_CHECKPOINT="$BASE_RUN/epoch-25"
SEED="${SEED:-42}"
RUN_DIR="$REPO/artifacts/lj_scales_des_production/seed_${SEED}"
CKPT_DIR="$RUN_DIR/ckpts"
cd "$REPO"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
mkdir -p "$RUN_DIR/logs" "$CKPT_DIR" "$REPO/artifacts/lj_scales_des_production/logs"

[[ -s "$DATASET" ]] || { echo "ERROR: missing $DATASET" >&2; exit 2; }
[[ -d "$BASE_CHECKPOINT" ]] || { echo "ERROR: missing warm start $BASE_CHECKPOINT" >&2; exit 2; }
echo "Production scaled-LJ fit: seed=$SEED sigma=[0.8,1.2] min_type_frames=25"
echo "Warm start: $BASE_CHECKPOINT (completed full-data fit, job 205839)"
python -c "import jax; print('JAX devices:', jax.devices())"

uv run mmml physnet-train \
  --config examples/lj_scales/train_des_full_production_scaled.yaml \
  --data "$DATASET" \
  --valid-data "$DATASET" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "hybrid_mm_scaled_lj_des_full_seed${SEED}" \
  --seed "$SEED" \
  --physnet-checkpoint "$BASE_CHECKPOINT"

LJ_DES=1 LJ_ARTIFACTS_DIR="$RUN_DIR" LJ_CKPT_DIR="$CKPT_DIR" \
  uv run python examples/lj_scales/06_inspect_scales.py
