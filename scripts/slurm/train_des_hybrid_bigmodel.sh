#!/bin/bash
# Higher-capacity DES hybrid-MM training, sweeping energy_weight.
#
# Capacity comes from examples/lj_scales/train_des_bigmodel.yaml (features 64,
# num_iterations 4, n_res 4, basis 32, use_energy_bias on, no warm start).
# The array varies only energy_weight, which is the one free parameter: the
# baseline's 1.0-vs-52.91 split made the loss ~98% forces and left energy 3.8x
# over target, but the right value is not derivable a priori.
#
# Each task writes its own tree; nothing here touches artifacts/lj_scales_des/.
#
#   sbatch scripts/slurm/train_des_hybrid_bigmodel.sh
#
#SBATCH --job-name=des-big
#SBATCH --partition=a100
#SBATCH --qos=a100-1day
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=artifacts/lj_scales_des_bigmodel_e/logs/slurm-%A_%a.out
#SBATCH --error=artifacts/lj_scales_des_bigmodel_e/logs/slurm-%A_%a.err

set -uo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO"

EW_LIST=(10 30 100)
EW="${EW_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

EPOCHS="${LJ_EPOCHS:-40}"
# Filtered set has 102,994 frames (108,500 - 5,506 close contacts), so the
# original 100000/8500 split no longer fits.
NTRAIN="${LJ_NTRAIN:-95000}"
NVALID="${LJ_NVALID:-7900}"

RUN_ROOT="$REPO/artifacts/lj_scales_des_bigmodel_e/ew_${EW}"
CKPT_DIR="$RUN_ROOT/ckpts"
# Close-contact filtered (<1.5 A removed: 5% of frames carrying 95% of the
# force MSE) AND per-element reference energies subtracted. The latter is
# essential without a warm start: composition explains 99.97% of raw E, so
# from scratch the model spent every epoch failing to rediscover ~800
# kcal/mol of offset (energy MAE flat at ~1200 for 6 epochs) while forces
# converged normally. Residual std is 12.8 kcal/mol.
DATA="${LJ_DATA:-$REPO/artifacts/lj_scales_des/des_dimers_cgenff_top50_min15_eref.npz}"

if [[ -e "$CKPT_DIR" ]]; then
  echo "REFUSING to run: $CKPT_DIR exists (would mix runs). Move it first." >&2
  exit 1
fi
mkdir -p "$CKPT_DIR"

# rich builds its Console with force_terminal=True, so without this the log
# fills with ANSI colour/cursor codes. TERM=dumb keeps the tables readable in
# a redirected file. (It does NOT make the Live table stream per-epoch --
# the plain "[epoch N/M]" line in training.py does that.)
export TERM=dumb

# Compute nodes have no outbound network; without these `uv run` re-resolves
# against pypi.org and the job dies on timeouts.
export UV_NO_SYNC=1
export UV_OFFLINE=1

export MMML_SCICORE_CMAKE="${MMML_SCICORE_CMAKE:-CMake/3.31.8-GCCcore-14.3.0}"
export MMML_SCICORE_TOOLCHAIN="${MMML_SCICORE_TOOLCHAIN:-foss/2025a}"
source scripts/scicore_env.sh
source .venv/bin/activate
set -e

echo "=== DES hybrid, high capacity ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "energy_weight=$EW  epochs=$EPOCHS  n_train=$NTRAIN  n_valid=$NVALID"
echo "data=$DATA"
echo "ckpt_dir=$CKPT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset $DATA -- run examples/lj_scales/12_des_dataset.sh first." >&2
  exit 1
fi

# NOTE: deliberately no --physnet-checkpoint. Warm starting would re-impose the
# checkpoint's features=32 / num_iterations=2 and silently undo the capacity
# increase this whole run exists to test.
uv run mmml physnet-train \
  --config examples/lj_scales/train_des_bigmodel.yaml \
  --data "$DATA" \
  --valid-data "" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "des_big_ew${EW}" \
  --n-train "$NTRAIN" \
  --n-valid "$NVALID" \
  --num-epochs "$EPOCHS" \
  --energy-weight "$EW"

echo "=== training done (energy_weight=$EW); scoring against target ==="

# Score in explicit kcal/mol (the training log's MAEs are eV) and report
# RMSE, which training never computes.
PARAMS=$(ls -t "$CKPT_DIR"/params_*.json 2>/dev/null | head -1)
SIDECAR=$(ls -t "$CKPT_DIR"/*/hybrid_mm.json 2>/dev/null | head -1)
if [[ -n "$PARAMS" && -n "$SIDECAR" ]]; then
  uv run python -m mmml.cli.misc.eval_hybrid_accuracy \
    --params "$PARAMS" \
    --hybrid-mm-json "$SIDECAR" \
    --data "$DATA" \
    --config examples/lj_scales/train_des_bigmodel.yaml \
    --n-train "$NTRAIN" --n-valid "$NVALID" --seed 42 \
    --json-out "$RUN_ROOT/accuracy.json" || true
else
  echo "WARNING: no params/sidecar found under $CKPT_DIR; skipping scoring." >&2
fi

echo "=== task complete (energy_weight=$EW) ==="
