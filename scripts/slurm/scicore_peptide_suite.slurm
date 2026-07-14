#!/usr/bin/env bash
#SBATCH --job-name=mmml_peptides_scicore
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=reports/slurm_peptides_%j.log
#SBATCH --error=reports/slurm_peptides_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Load sciCORE environment
source "${REPO_ROOT}/scripts/env/scicore.env"

echo "[SciCORE Slurm] Running Peptides Suite (trialanine, alanine)..."
mkdir -p reports/scicore_peptides

# Run orchestration workflow for peptides on scicore
"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env scicore \
  --category peptides \
  --systems trialanine alanine \
  --output-dir reports/scicore_peptides

# Generate proof-of-work report
"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/scicore_peptides \
  --report-out reports/pow_peptides_scicore.md

echo "[SciCORE Slurm] Peptides Suite Complete."
