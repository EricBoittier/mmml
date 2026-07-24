#!/usr/bin/env bash
# Mechanical-embedding smokes: Packmol composition (ACN/TIP3/DMSO) × ASE/JAX-MD/PyCHARMM.
# Optionally build make-box artifacts and run from-pdb campaigns.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

RUN_MAKE_BOX="${RUN_MAKE_BOX:-1}"
RUN_FROM_BOX="${RUN_FROM_BOX:-1}"
RUN_PYCHARMM="${RUN_PYCHARMM:-1}"
SOLVENTS="${SOLVENTS:-tip3 acn dmso}"

echo "=== export solute PDB ==="
uv run python examples/m/07_export_solute_pdb.py

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi

if [[ "${RUN_MAKE_BOX}" == "1" ]]; then
  bash examples/m/08_make_boxes.sh
else
  echo "Skipping make-box (RUN_MAKE_BOX=${RUN_MAKE_BOX})"
fi

run_campaign() {
  local cfg="$1"
  local need_charmm="${2:-1}"
  if [[ "${need_charmm}" == "1" && "${has_pycharmm}" != "1" ]]; then
    echo "SKIP (no PyCHARMM): ${cfg}"
    return 0
  fi
  if [[ "${need_charmm}" == "1" && "${RUN_PYCHARMM}" != "1" ]]; then
    # Still run ASE / jaxmd jobs by filtering? Campaign --run-all needs CHARMM for PSF.
    echo "SKIP (RUN_PYCHARMM=0): ${cfg}"
    return 0
  fi
  echo "=== md-system --config ${cfg} --run-all ==="
  uv run mmml md-system --config "${cfg}" --run-all
}

for sol in ${SOLVENTS}; do
  run_campaign "examples/m/yaml/mech_embed_${sol}.yaml" 1
done

if [[ "${RUN_FROM_BOX}" == "1" ]]; then
  for sol in ${SOLVENTS}; do
    pdb="${ARTIFACTS_DIR}/boxes/${sol}/model.pdb"
    if [[ ! -f "${pdb}" ]]; then
      echo "SKIP from-box ${sol}: missing ${pdb} (run 08_make_boxes.sh)"
      continue
    fi
    run_campaign "examples/m/yaml/mech_embed_from_box_${sol}.yaml" 1
  done
fi

echo "PASS: mechanical-embedding smokes (artifacts under ${ARTIFACTS_DIR})"
