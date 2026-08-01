#!/bin/bash
# Full-data demo of the opt-in additive shared-cutoff Hamiltonian.
# Existing handoff jobs are untouched; fixed LJ keeps this first demo
# identifiable before any later stage releases per-type LJ scales.
#SBATCH --job-name=des-shared-rc6
#SBATCH --partition=gpu
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_shared_cutoff/logs/slurm-%j.out
#SBATCH --error=/mmhome/boittier/home/mmml/artifacts/lj_scales_des_shared_cutoff/logs/slurm-%j.err

set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml}"
DATASET="$REPO/artifacts/lj_scales_des/des_dimers_cgenff_all.npz"
RUN_DIR="$REPO/artifacts/lj_scales_des_shared_cutoff"
CKPT_DIR="$RUN_DIR/ckpts"
cd "$REPO"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
mkdir -p "$RUN_DIR/logs" "$CKPT_DIR"
test -s "$DATASET"

echo "Hamiltonian: shared_cutoff (additive ML + force-shifted MM)"
echo "Shared atomic cutoff: 6.0 A"
echo "Dataset: $DATASET (full train + in-sample evaluation)"
echo "LJ scales: fixed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run mmml physnet-train \
  --config examples/hybrid_mm_charges/train_shared_cutoff.yaml \
  --data "$DATASET" \
  --valid-data "$DATASET" \
  --ckpt-dir "$CKPT_DIR" \
  --tag des_shared_cutoff_rc6_fixed_lj \
  --num-epochs 25 \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json

echo "=== shared-cutoff demo training complete ==="
