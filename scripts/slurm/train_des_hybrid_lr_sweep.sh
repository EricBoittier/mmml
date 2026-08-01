#!/bin/bash
# LR sweep for the DES hybrid-MM warm start (diagnosing job 19360535).
#
# 19360535 completed cleanly but did not converge: valid loss moved 0.3% over
# its last 10 epochs while the *raw* train loss oscillated 745-829 with no
# trend. Train loss is measured on `params` and valid loss on `ema_params`, so
# that gap is params-vs-EMA, not overfitting -- the signal that matters is the
# flat, noisy raw-train curve, i.e. lr=1e-3 is too high for this loss surface.
#
# Array index selects the learning rate. Index 0 re-runs 1e-3 as a control, so
# "the sweep improved things" can be read against a same-length baseline rather
# than against the 25-epoch original.
#
# Each task writes to its own artifacts/ckpt directory, so nothing here can
# overwrite the 19360535 outputs under artifacts/lj_scales_des/.
#
#   sbatch scripts/slurm/train_des_hybrid_lr_sweep.sh
#
#SBATCH --job-name=des-ws-lr
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=artifacts/lj_scales_des_lrsweep/logs/slurm-%A_%a.out
#SBATCH --error=artifacts/lj_scales_des_lrsweep/logs/slurm-%A_%a.err

set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO"

LRS=(0.001 0.0003 0.0001)
LR="${LRS[${SLURM_ARRAY_TASK_ID:-0}]}"

EPOCHS="${LJ_EPOCHS:-10}"
NTRAIN="${LJ_NTRAIN:-100000}"
NVALID="${LJ_NVALID:-8500}"

# Separate tree per task: never write into artifacts/lj_scales_des/.
SWEEP_ROOT="$REPO/artifacts/lj_scales_des_lrsweep/lr_${LR}"
CKPT_DIR="$SWEEP_ROOT/ckpts"
DATA="$REPO/artifacts/lj_scales_des/des_dimers_cgenff_top50.npz"

if [[ -e "$CKPT_DIR" ]]; then
  echo "REFUSING to run: $CKPT_DIR already exists (would mix runs)." >&2
  echo "Move or delete it first." >&2
  exit 1
fi
mkdir -p "$CKPT_DIR"

# Compute nodes have no outbound network; without these every `uv run`
# re-resolves the editable install against pypi.org and the job dies on
# timeouts. The venv is prebuilt, so no sync is needed at run time.
export UV_NO_SYNC=1
export UV_OFFLINE=1

export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
source scripts/scicore_env.sh
source .venv/bin/activate
set -e

echo "=== DES warm-start LR sweep ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "learning_rate=$LR  epochs=$EPOCHS  n_train=$NTRAIN  n_valid=$NVALID"
echo "data=$DATA"
echo "ckpt_dir=$CKPT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset $DATA -- run examples/lj_scales/12_des_dataset.sh first." >&2
  exit 1
fi

uv run mmml physnet-train \
  --config examples/lj_scales/train_des_warmstart.yaml \
  --data "$DATA" \
  --valid-data "" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "des_ws_lr${LR}" \
  --n-train "$NTRAIN" \
  --n-valid "$NVALID" \
  --num-epochs "$EPOCHS" \
  --learning-rate "$LR" \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json

echo "=== LR sweep task complete (lr=$LR) ==="
