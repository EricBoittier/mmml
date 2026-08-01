#!/bin/bash
# Collect a completed ORCA array into reproducible train/valid/test NPZ files.
# Submit with --dependency=afterok:<array-job-id>.
#SBATCH --job-name=orca-lj-collect
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G

set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml}"
ARTIFACTS="${LJ_ARTIFACTS_DIR:?LJ_ARTIFACTS_DIR is required}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export UV_NO_SYNC=1
source .venv/bin/activate

LJ_JOINT=1 \
LJ_ARTIFACTS_DIR="$ARTIFACTS" \
LJ_ORCA_MODE=collect \
  bash examples/lj_scales/09_submit_orca_rimp2.sh

for split in train valid test; do
  file="$ARTIFACTS/joint_rimp2_${split}.npz"
  [[ -s "$file" ]] || { echo "ERROR: missing collected $file" >&2; exit 2; }
done
echo "ORCA collection complete: $ARTIFACTS/joint_rimp2_{train,valid,test}.npz"
