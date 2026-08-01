#!/bin/bash
# Prepare DES hybrid-MM data and warm-start training on pcstudix.
#SBATCH --job-name=des-hybrid-ws
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/lj_scales_des/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/lj_scales_des/logs/slurm-%j.err

set -euo pipefail

REPO="${MMML_REPO:-$HOME/mmml}"
cd "$REPO"
source .venv/bin/activate

export PATH="$HOME/.local/bin:$PATH"
export LJ_DES=1
export LJ_DEVICE=gpu
export LJ_DES_H5="${LJ_DES_H5:-$HOME/qcell_dimers.h5}"
export LJ_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export LJ_DES_TOP_RESIDUES="${LJ_DES_TOP_RESIDUES:-50}"
export LJ_EPOCHS="${LJ_EPOCHS:-25}"
export LJ_NTRAIN="${LJ_NTRAIN:-100000}"
export LJ_NVALID="${LJ_NVALID:-10000}"

source examples/lj_scales/_env.sh
mkdir -p "$LJ_ARTIFACTS_DIR" "$LJ_CKPT_DIR"

echo "=== DES hybrid warm start on pcstudix ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "source=$LJ_DES_H5"
echo "dataset=$LJ_ENRICHED"
echo "checkpoint=$REPO/examples/ckpts_json/DESdimers_params.json"
echo "epochs=$LJ_EPOCHS n_train=$LJ_NTRAIN n_valid=$LJ_NVALID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('JAX devices:', jax.devices())"

bash examples/lj_scales/12_des_dataset.sh

uv run mmml physnet-train \
  --config examples/lj_scales/train_des_warmstart.yaml \
  --data "$LJ_ENRICHED" \
  --valid-data "" \
  --ckpt-dir "$LJ_CKPT_DIR" \
  --tag "$LJ_TAG" \
  --n-train "$LJ_NTRAIN" \
  --n-valid "$LJ_NVALID" \
  --num-epochs "$LJ_EPOCHS" \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json

uv run python examples/lj_scales/06_inspect_scales.py
echo "=== DES hybrid warm start complete ==="
