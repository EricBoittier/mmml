#!/usr/bin/env bash
# DMC vibrational ground-state estimates at reactant / product basins (gas phase).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

NSTEP="${NSTEP:-100}"
EQSTEP="${EQSTEP:-20}"
NWALKER="${NWALKER:-32}"

echo "=== export basin endpoints ==="
if [[ ! -f examples/m/neb/reag_0_opt.xyz || ! -f examples/m/neb/prod_0_opt.xyz ]]; then
  uv run python examples/m/07_export_neb_endpoints.py
fi

run_dmc() {
  local label="$1"
  local xyz="$2"
  local out="${ARTIFACTS_DIR}/dmc_${label}"
  echo "=== DMC ${label}: ${xyz} ==="
  uv run mmml dmc \
    --natm 9 \
    --nwalker "${NWALKER}" \
    --stepsize 5e-4 \
    --nstep "${NSTEP}" \
    --eqstep "${EQSTEP}" \
    --alpha 1200.0 \
    --max-batch "${NWALKER}" \
    --seed 0 \
    --checkpoint "${MMML_CKPT}" \
    --input "${xyz}" \
    --output-dir "${out}"
  local log
  log="$(find "${out}" -maxdepth 1 -name '*.log' | head -1)"
  if [[ -z "${log}" ]]; then
    echo "FAIL: no DMC log under ${out}"
    exit 1
  fi
  echo "PASS: DMC ${label} -> ${out} (${log})"
}

run_dmc react examples/m/neb/reag_0_opt.xyz
run_dmc product examples/m/neb/prod_0_opt.xyz

echo "PASS: DMC react + product basins under ${ARTIFACTS_DIR}/dmc_*"
