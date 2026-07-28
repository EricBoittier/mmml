#!/usr/bin/env bash
# Fixed-bias umbrella-sample smoke (gas phase, PhysNet + JAX-MD).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/umbrella_nc_gas.yaml}"
OUT="${ARTIFACTS_DIR}/umbrella_nc_gas"

echo "=== export NEB endpoints (umbrella structure) ==="
if [[ ! -f examples/m/neb/reag_0_opt.xyz ]]; then
  uv run python examples/m/07_export_neb_endpoints.py
fi

echo "=== umbrella-sample: $(basename "${CFG}") ==="
uv run mmml umbrella-sample --config "${CFG}" --output-dir "${OUT}" --overwrite

SUMMARY="${OUT}/umbrella_summary.json"
SNAP="${OUT}/umbrella_snapshots.npz"
for f in "${SUMMARY}" "${SNAP}"; do
  if [[ ! -f "${f}" ]]; then
    echo "FAIL: missing ${f}"
    exit 1
  fi
done

echo "PASS: umbrella-sample gas -> ${OUT}"
