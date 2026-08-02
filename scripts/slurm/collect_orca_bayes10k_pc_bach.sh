#!/bin/bash
# Collect the descriptor-designed 10k RI-MP2 campaign after its array succeeds.
#SBATCH --job-name=orca-bayes10k-collect
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G

set -euo pipefail
REPO="${MMML_REPO:-$HOME/mmml}"
CAMPAIGN="${LJ_BAYES10K_DIR:-$REPO/artifacts/lj_scales_bayes_10k}"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH" UV_NO_SYNC=1
source .venv/bin/activate

python scripts/collect_orca_array.py \
  --run-dir "$CAMPAIGN/orca_rimp2" \
  --source "$CAMPAIGN/selected_10000.npz" \
  --out "$CAMPAIGN/selected_10000_rimp2.npz" \
  --method "RI-MP2/def2-TZVP"

for split in train valid test; do
  file="$CAMPAIGN/selected_10000_rimp2_${split}.npz"
  [[ -s "$file" ]] || { echo "ERROR: missing collected $file" >&2; exit 2; }
done
echo "RI-MP2 Bayes-10k collection complete"
