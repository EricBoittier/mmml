#!/usr/bin/env bash
#SBATCH --job-name=mmml_liquids_scicore
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=reports/slurm_liquid_%j.log
#SBATCH --error=reports/slurm_liquid_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

# Load sciCORE environment
source "${REPO_ROOT}/scripts/env/scicore.env"

echo "[SciCORE Slurm] Running Condensed Phase Liquids Suite (BENZ, TIP3, DCM, ACO)..."
mkdir -p reports/scicore_liquids

# Run orchestration workflow for liquids on scicore
"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env scicore \
  --category liquids \
  --systems BENZ TIP3 DCM ACO \
  --output-dir reports/scicore_liquids

# Generate proof-of-work report
"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/scicore_liquids \
  --report-out reports/pow_condensed_phase_liquids_scicore.md

echo "[SciCORE Slurm] Condensed Phase Liquids Suite Complete."
