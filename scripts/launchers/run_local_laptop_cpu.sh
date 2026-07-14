#!/usr/bin/env bash
# Local CPU runner for local_laptop (macOS / standard debugging)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/env/local_laptop.env"

echo "[local_laptop Launcher] Running standard CPU validation and report generation..."
mkdir -p reports/local_laptop_cpu

"${MMML_PYTHON}" scripts/orchestrate_goals.py \
  --env local_laptop \
  --category all \
  --systems BENZ TIP3 DCM ACO trialanine alanine \
  --output-dir reports/local_laptop_cpu

"${MMML_PYTHON}" scripts/generate_proof_of_work.py \
  --input-dir reports/local_laptop_cpu \
  --report-out reports/pow_local_laptop_cpu.md

echo "[local_laptop Launcher] Local Laptop Run Completed."
