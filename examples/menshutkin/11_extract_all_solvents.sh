#!/usr/bin/env bash
# Extract CGenFF parameters for every campaign solvent (one subprocess each,
# because CHARMM only builds one system per process).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/examples/menshutkin/_env.sh"
cd "${ROOT}"

# name:residue:density(kg/m3 at 298 K):box side(A), following Turan et al.
for entry in methanol:MEOH:792:25 acetonitrile:ACN:786:28 benzene:BENZ:874:27 cyclohexane:CHEX:774:30; do
  IFS=: read -r name resi rho side <<<"${entry}"
  echo "=== ${name} (${resi}) ==="
  uv run python examples/menshutkin/10_extract_solvent_params.py \
    --residue "${resi}" --name "${name}" --density "${rho}" --box-side "${side}" \
    2>&1 | grep -E "^${resi}:|net charge|bonds,|Wrote|FAIL" || echo "  FAILED"
done
