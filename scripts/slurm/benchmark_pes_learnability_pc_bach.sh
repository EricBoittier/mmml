#!/bin/bash
# Run after the dependent ORCA collector has written RI-MP2 train/test NPZs.
#SBATCH --job-name=pes-learnability
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml}"
ARTIFACTS="${LJ_ARTIFACTS_DIR:?LJ_ARTIFACTS_DIR is required}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
source .venv/bin/activate
python scripts/benchmark_pes_design_learnability.py \
  --train "$ARTIFACTS/joint_rimp2_train.npz" \
  --test "$ARTIFACTS/joint_rimp2_test.npz" \
  --out-dir "$ARTIFACTS/learnability_benchmark" \
  --sizes 50,100,200,500,1000 \
  --repeats 5
