#!/usr/bin/env bash
# Full NH3–CH3Cl report pipeline: evaluate → MD smokes → docs figures/report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

bash examples/m/01_evaluate.sh
bash examples/m/run_md_smokes.sh
uv run python examples/m/02_figures_and_report.py

echo "PASS: examples/m run_all -> docs/examples/nh3-ch3cl-results.md"
