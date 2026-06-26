#!/usr/bin/env bash
# Run CPU MD example ladder (no CUDA). PyCHARMM md-system steps are optional.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_cpu/_env.sh"
cd "${ROOT}"
DIR="examples/md_cpu"

echo "=== CPU MD examples (md-cpu extra) ==="
uv run python "${DIR}/00_check_env.py"
bash "${DIR}/01_neighbor_list_parity.sh"
uv run python "${DIR}/02_ml_energy_ase.py"
uv run python "${DIR}/03_ml_energy_jaxmd.py"
uv run python "${DIR}/05_free_nve_ase_smoke.py"
uv run python "${DIR}/06_free_nve_jaxmd_smoke.py"
bash "${DIR}/07_nl_backend_matrix.sh"

if [[ -n "${CHARMM_LIB_DIR:-}" ]] && uv run python -c "import pycharmm" 2>/dev/null; then
  echo "=== PyCHARMM md-system examples ==="
  bash "${DIR}/04_md_system_evaluate_ase.sh"
  bash "${DIR}/05_md_system_free_nve_ase.sh"
  bash "${DIR}/06_md_system_free_nve_jaxmd.sh"
else
  echo "SKIP: md-system examples (04, 05/06 shell) — PyCHARMM not configured"
fi

echo "=== ALL CPU MD EXAMPLES PASSED ==="
