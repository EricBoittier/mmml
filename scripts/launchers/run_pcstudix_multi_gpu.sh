#!/usr/bin/env bash
# Local multi-GPU runner for pcstudix (4x RTX 5090 GPUs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/env/pcstudix.env"

echo "[pcstudix Launcher] Dispatching jobs across available RTX 5090 GPUs..."
mkdir -p reports/pcstudix_multi_gpu

# Execute parallel orchestration on pcstudix
"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env pcstudix \
  --category all \
  --systems BENZ TIP3 DCM ACO trialanine alanine \
  --output-dir reports/pcstudix_multi_gpu

# Generate proof-of-work report
"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/pcstudix_multi_gpu \
  --report-out reports/pow_pcstudix_multi_gpu.md

echo "[pcstudix Launcher] High-throughput Multi-GPU Run Completed."
