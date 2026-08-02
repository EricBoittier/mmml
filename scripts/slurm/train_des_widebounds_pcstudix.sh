#!/bin/bash
# Bound-saturation diagnostic. Identical to the production scaled-LJ fit except
# the sigma/epsilon scale bounds are opened far wider.
#
# Why: every completed warm-started fit pins ~29% of its sigma scales AT the
# configured limits (0.95/1.05 or 0.8/1.2) and 14-19% of its epsilon scales at
# 0.25/4.0. A scale sitting on its bound is not a measurement, it is the
# optimizer being stopped. The ZBL hypothesis is already refuted -- des-nozbl
# reproduces the baseline saturation exactly -- so the remaining suspect is the
# truncated-MIC Coulomb (lr_solver: mic) being absorbed into sigma.
#
# Read it as:
#   scales settle INSIDE the new bounds -> the old bounds were merely tight
#   scales run to the NEW bounds        -> systematic error is being absorbed
#                                          and no bound setting rescues it
#
# Bounds are passed as CLI flags rather than a YAML: untracked files on this
# machine do not survive the concurrent branch switches happening in this
# checkout, and a missing --config killed the first attempt (job 205877).
#SBATCH --job-name=des-widebounds
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_widebounds/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_widebounds/logs/slurm-%j.err

set -euo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
DATASET="$REPO/artifacts/lj_scales_des/des_dimers_cgenff_all.npz"
BASE_RUN="$REPO/artifacts/lj_scales_des_full/ckpts/hybrid_mm_fixed_lj_scales_des_full_insample-155c22fe-5788-42c0-9dc2-fcf04ffdd049"
BASE_CHECKPOINT="$BASE_RUN/epoch-25"
SEED="${SEED:-42}"
RUN_DIR="$REPO/artifacts/lj_scales_des_widebounds"
CKPT_DIR="$RUN_DIR/ckpts"
CONFIG="$REPO/examples/lj_scales/train_des_full_production_scaled.yaml"
cd "$REPO"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
export UV_NO_SYNC=1
export UV_OFFLINE=1
mkdir -p "$RUN_DIR/logs" "$CKPT_DIR"

[[ -s "$DATASET" ]]        || { echo "ERROR: missing dataset $DATASET" >&2; exit 2; }
[[ -d "$BASE_CHECKPOINT" ]]|| { echo "ERROR: missing warm start $BASE_CHECKPOINT" >&2; exit 2; }
[[ -s "$CONFIG" ]]         || { echo "ERROR: missing config $CONFIG" >&2; exit 2; }

echo "Bound-saturation diagnostic: seed=$SEED sigma=[0.6,1.6] eps=[0.05,20.0]"
echo "Warm start: $BASE_CHECKPOINT"
python -c "import jax; print('JAX devices:', jax.devices())"

# Disjoint hold-out: --valid-data  made validation in-sample, and
# with objective=valid_loss + best=true the saved epoch was then selected on
# data the model had already trained on. Empty --valid-data makes
# physnet-train split --data itself (0-overlap verified on des-hybrid-ws).
uv run mmml physnet-train \
  --config "$CONFIG" \
  --data "$DATASET" \
  --valid-data "" \
  --n-train 108000 \
  --n-valid 12000 \
  --ckpt-dir "$CKPT_DIR" \
  --tag "hybrid_mm_widebounds_des_seed${SEED}" \
  --seed "$SEED" \
  --mm-lj-sigma-scale-min 0.6 \
  --mm-lj-sigma-scale-max 1.6 \
  --mm-lj-epsilon-scale-min 0.05 \
  --mm-lj-epsilon-scale-max 20.0 \
  --physnet-checkpoint "$BASE_CHECKPOINT"

LJ_DES=1 LJ_ARTIFACTS_DIR="$RUN_DIR" LJ_CKPT_DIR="$CKPT_DIR" \
  uv run python examples/lj_scales/06_inspect_scales.py
