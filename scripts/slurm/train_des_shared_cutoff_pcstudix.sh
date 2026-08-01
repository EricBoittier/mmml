#!/bin/bash
# Full-data demo of the opt-in additive shared-cutoff Hamiltonian.
# Existing handoff jobs are untouched; fixed LJ keeps this first demo
# identifiable before any later stage releases per-type LJ scales.
#SBATCH --job-name=des-shared
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
RC="${SHARED_CUTOFF:-6}"
RC_TAG="${RC//./p}"
RUN_DIR="$REPO/artifacts/lj_scales_des_shared_cutoff/rc${RC_TAG}"
CKPT_DIR="$RUN_DIR/ckpts"
cd "$REPO"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
mkdir -p "$RUN_DIR/logs" "$CKPT_DIR"
test -s "$DATASET"

echo "Hamiltonian: shared_cutoff (additive ML + force-shifted MM)"
echo "Shared atomic cutoff: $RC A (not COM distance)"
echo "Dataset: $DATASET (full train + in-sample evaluation)"
echo "LJ scales: fixed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run mmml physnet-train \
  --config examples/hybrid_mm_charges/train_shared_cutoff_des_transfer.yaml \
  --data "$DATASET" \
  --valid-data "$DATASET" \
  --ckpt-dir "$CKPT_DIR" \
  --tag "des_shared_cutoff_rc${RC_TAG}_fixed_lj" \
  --num-epochs 25 \
  --cutoff "$RC" \
  --shared-cutoff "$RC" \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json \
  --no-match-checkpoint-architecture

echo "=== shared-cutoff demo training complete ==="
