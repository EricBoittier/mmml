#!/usr/bin/env bash
# Launch the gas-phase PMF on the GPU host.
#
#   ssh gpu09 /mmhome/andreychev/mmml/mmml/examples/menshutkin/run_gas_gpu.sh
#
# Writes a log next to the artifacts so progress can be tailed from anywhere.
set -euo pipefail

ROOT=/mmhome/andreychev/mmml/mmml
cd "${ROOT}"
# shellcheck source=/dev/null
source "${ROOT}/examples/menshutkin/_env.sh"

LOG="${MENSH_ARTIFACTS}/gas/gas_pmf.log"
mkdir -p "$(dirname "${LOG}")"

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee "${LOG}"
"${ROOT}/.venv/bin/python" -c "import jax; print('jax devices:', jax.devices())" 2>&1 \
  | grep -v '^ ' | tee -a "${LOG}"

bash "${ROOT}/examples/menshutkin/02_gas_pmf.sh" 2>&1 | tee -a "${LOG}"
