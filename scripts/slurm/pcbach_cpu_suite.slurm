#!/usr/bin/env bash
#SBATCH --job-name=mmml_pcbach_cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=reports/slurm_pcbach_%j.log
#SBATCH --error=reports/slurm_pcbach_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Load pc-bach environment
source "${REPO_ROOT}/scripts/env/pcbach.env"

echo "[pc-bach Slurm] Running High-Core CPU Suite (MPI / PyCHARMM / Liquids & Peptides)..."
mkdir -p reports/pcbach_cpu

# Run orchestration workflow for CPU jobs on pcbach
"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env pcbach \
  --category all \
  --systems BENZ TIP3 DCM ACO trialanine alanine \
  --output-dir reports/pcbach_cpu

# Generate proof-of-work report
"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/pcbach_cpu \
  --report-out reports/pow_pcbach_cpu.md

echo "[pc-bach Slurm] High-Core CPU Suite Complete."
