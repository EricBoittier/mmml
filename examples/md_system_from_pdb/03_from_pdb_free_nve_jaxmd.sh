#!/usr/bin/env bash
# Full-system PDB → JAX-MD vacuum NVE smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/03_from_pdb_free_nve_jaxmd.yaml"
OUT="${ARTIFACTS_DIR}/03_free_nve_jaxmd"

echo "=== config $(basename "${CFG}") ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-pdb "${PDB_MONOMER}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: jaxmd NVE smoke -> ${OUT}"
