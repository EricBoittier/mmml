#!/usr/bin/env bash
# Run smoke examples; skip PyCHARMM / certified steps when unavailable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

have_pycharmm=0
if uv run python -c "import pycharmm" 2>/dev/null; then
  have_pycharmm=1
fi

run_step() {
  local script="$1"
  local need_charmm="${2:-0}"
  echo
  echo "######## ${script} ########"
  if [[ "${need_charmm}" == "1" && "${have_pycharmm}" != "1" ]]; then
    echo "SKIP: PyCHARMM not importable"
    return 0
  fi
  bash "${ROOT}/examples/md_system_from_pdb/${script}"
}

# ASE / jaxmd from-pdb (need CHARMM for PSF build in hybrid path on some setups;
# keep them in the CHARMM-gated group if import fails — still try ASE first).
run_step 02_from_pdb_free_nve_ase.sh 1
run_step 03_from_pdb_free_nve_jaxmd.sh 1
run_step 01_from_pdb_pycharmm_minimize.sh 1
run_step 04_from_pdb_free_nve_pycharmm.sh 1
run_step 05_packmol_mix_pdb_monomer.sh 1
run_step 06_from_pdb_nvt_fix_resids.sh 1
run_step 07_certified_psf_crd_pbc.sh 0

echo
echo "Done. Artifacts under ${ARTIFACTS_DIR}"
