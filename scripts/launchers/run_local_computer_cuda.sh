#!/usr/bin/env bash
# Local CUDA runner for local_computer (RTX 4060 GPU debugging)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/env/local_computer.env"

echo "[local_computer Launcher] Running CUDA integration and debugging suite..."
mkdir -p reports/local_computer_cuda

"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env local_computer \
  --category all \
  --systems BENZ TIP3 DCM ACO trialanine alanine \
  --output-dir reports/local_computer_cuda

"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/local_computer_cuda \
  --report-out reports/pow_local_computer_cuda.md

echo "[local_computer Launcher] Local CUDA Run Completed."
