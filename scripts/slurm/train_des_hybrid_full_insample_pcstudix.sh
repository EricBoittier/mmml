#!/bin/bash
# Full-data DES fit on pcstudix. The same complete CGenFF-assigned NPZ is used
# for training and epoch evaluation; thermodynamic properties are the external
# validation target, so no frames are withheld here.
#SBATCH --job-name=des-full-fit
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_full/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_full/logs/slurm-%j.err

set -euo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
DATASET="$REPO/artifacts/lj_scales_des/des_dimers_cgenff_all.npz"
RUN_DIR="$REPO/artifacts/lj_scales_des_full"
CKPT_DIR="$RUN_DIR/ckpts"
cd "$REPO"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
mkdir -p "$RUN_DIR/logs" "$CKPT_DIR"

if [[ ! -s "$DATASET" ]]; then
  echo "ERROR: complete CGenFF-assigned dataset is missing: $DATASET" >&2
  exit 2
fi

N_FRAMES="$(python - "$DATASET" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as data:
    print(len(data["E"]))
PY
)"

echo "=== Full-data DES in-sample fit on pcstudix ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "dataset=$DATASET"
echo "frames=$N_FRAMES (all used for training and evaluation)"
echo "checkpoint=$REPO/examples/ckpts_json/DESdimers_params.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices())"

uv run mmml physnet-train \
  --config examples/lj_scales/train_des_full_insample.yaml \
  --data "$DATASET" \
  --valid-data "$DATASET" \
  --ckpt-dir "$CKPT_DIR" \
  --tag hybrid_mm_fixed_lj_scales_des_full_insample \
  --num-epochs 25 \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json

LJ_DES=1 LJ_ARTIFACTS_DIR="$RUN_DIR" LJ_CKPT_DIR="$CKPT_DIR" \
  uv run python examples/lj_scales/06_inspect_scales.py
echo "=== Full-data DES in-sample fit complete ==="
