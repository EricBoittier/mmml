#!/usr/bin/env bash
# Full-system PDB → ASE vacuum NVE smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/02_from_pdb_free_nve_ase.yaml"
OUT="${ARTIFACTS_DIR}/02_free_nve_ase"

echo "=== config $(basename "${CFG}") ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-pdb "${PDB_MONOMER}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: ASE NVE smoke -> ${OUT}"
