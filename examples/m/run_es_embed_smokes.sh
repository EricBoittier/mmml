#!/usr/bin/env bash
# Electrostatic-embedding smokes (q0 / latent / latent_dynamic) × ASE/JAX-MD/PyCHARMM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

RUN_FROM_BOX="${RUN_FROM_BOX:-0}"
RUN_PYCHARMM="${RUN_PYCHARMM:-1}"
SOLVENTS="${SOLVENTS:-tip3 acn dmso}"
RUN_LATENT="${RUN_LATENT:-1}"
RUN_LATENT_DYNAMIC="${RUN_LATENT_DYNAMIC:-1}"
RUN_Q0_EWALD="${RUN_Q0_EWALD:-1}"

has_pycharmm=0
if uv run python -c "import pycharmm" >/dev/null 2>&1; then
  has_pycharmm=1
fi

run_campaign() {
  local cfg="$1"
  if [[ "${has_pycharmm}" != "1" ]]; then
    echo "SKIP (no PyCHARMM): ${cfg}"
    return 0
  fi
  if [[ "${RUN_PYCHARMM}" != "1" ]]; then
    echo "SKIP (RUN_PYCHARMM=0): ${cfg}"
    return 0
  fi
  echo "=== md-system --config ${cfg} --run-all ==="
  uv run mmml md-system --config "${cfg}" --run-all
}

for sol in ${SOLVENTS}; do
  run_campaign "examples/m/yaml/es_embed_${sol}.yaml"
done

if [[ "${RUN_LATENT}" == "1" ]]; then
  run_campaign examples/m/yaml/es_embed_dimer_latent.yaml
fi
if [[ "${RUN_LATENT_DYNAMIC}" == "1" ]]; then
  run_campaign examples/m/yaml/es_embed_tip3_latent_dynamic.yaml
fi
if [[ "${RUN_Q0_EWALD}" == "1" ]]; then
  run_campaign examples/m/yaml/es_embed_tip3_ewald.yaml
fi

if [[ "${RUN_FROM_BOX}" == "1" ]]; then
  pdb="${ARTIFACTS_DIR}/boxes/tip3/model.pdb"
  if [[ -f "${pdb}" ]]; then
    run_campaign examples/m/yaml/es_embed_from_box_tip3.yaml
  else
    echo "SKIP from-box: missing ${pdb} (run 08_make_boxes.sh)"
  fi
fi

echo "PASS: electrostatic-embedding smokes (artifacts under ${ARTIFACTS_DIR})"
