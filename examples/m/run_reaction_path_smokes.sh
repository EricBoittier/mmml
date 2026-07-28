#!/usr/bin/env bash
# Reaction-path toolkit smokes: umbrella-sample (gas) + NEB + DMC basins.
# Solvated adaptive umbrellas: 09_adumb_nc_distance.sh with SOLVATED=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

RUN_UMBRELLA="${RUN_UMBRELLA:-1}"
RUN_NEB="${RUN_NEB:-1}"
RUN_DMC="${RUN_DMC:-1}"
RUN_ADUMB="${RUN_ADUMB:-0}"

if [[ "${RUN_UMBRELLA}" == "1" ]]; then
  bash examples/m/14_umbrella_sample_gas.sh
fi
if [[ "${RUN_NEB}" == "1" ]]; then
  bash examples/m/13_neb.sh
fi
if [[ "${RUN_DMC}" == "1" ]]; then
  bash examples/m/15_dmc_basins.sh
fi
if [[ "${RUN_ADUMB}" == "1" ]]; then
  USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh
  SOLVATED=1 USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh
fi

echo "PASS: reaction-path smokes (artifacts under ${ARTIFACTS_DIR})"
