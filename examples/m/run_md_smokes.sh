#!/usr/bin/env bash
# Free-space NVE/NVT smokes: ML-only ASE/JAX-MD scripts + md-system (3 backends).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

N_STEPS="${N_STEPS:-40}"
RUN_MD_SYSTEM="${RUN_MD_SYSTEM:-1}"
RUN_PYCHARMM="${RUN_PYCHARMM:-1}"

echo "=== ML-only ASE / JAX-MD (dataset dimer frame) ==="
uv run python examples/m/03_free_nve_ase.py --n-steps "${N_STEPS}"
uv run python examples/m/04_free_nvt_ase.py --n-steps "${N_STEPS}"
uv run python examples/m/05_free_nve_jaxmd.py --n-steps "${N_STEPS}"
uv run python examples/m/06_free_nvt_jaxmd.py --n-steps "${N_STEPS}"

if [[ "${RUN_MD_SYSTEM}" != "1" ]]; then
  echo "Skipping md-system (RUN_MD_SYSTEM=${RUN_MD_SYSTEM})"
  exit 0
fi

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi

run_yaml() {
  local cfg="$1"
  local need_charmm="${2:-0}"
  if [[ "${need_charmm}" == "1" && "${has_pycharmm}" != "1" ]]; then
    echo "SKIP (no PyCHARMM): ${cfg}"
    return 0
  fi
  if [[ "${need_charmm}" == "1" && "${RUN_PYCHARMM}" != "1" ]]; then
    echo "SKIP (RUN_PYCHARMM=0): ${cfg}"
    return 0
  fi
  echo "=== md-system --config ${cfg} ==="
  uv run mmml md-system --config "${cfg}"
}

run_yaml examples/m/yaml/free_nve_ase.yaml 1
run_yaml examples/m/yaml/free_nvt_ase.yaml 1
run_yaml examples/m/yaml/free_nve_jaxmd.yaml 1
run_yaml examples/m/yaml/free_nvt_jaxmd.yaml 1
run_yaml examples/m/yaml/free_nve_pycharmm.yaml 1
run_yaml examples/m/yaml/free_nvt_pycharmm.yaml 1

echo "PASS: MD smokes (artifacts under ${ARTIFACTS_DIR})"
