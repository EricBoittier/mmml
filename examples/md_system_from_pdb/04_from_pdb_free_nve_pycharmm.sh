#!/usr/bin/env bash
# Full-system PDB → PyCHARMM vacuum NVE with flat-bottom restraint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/04_from_pdb_free_nve_pycharmm.yaml"
OUT="${ARTIFACTS_DIR}/04_free_nve_pycharmm"

echo "=== config $(basename "${CFG}") ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-pdb "${PDB_MONOMER}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: pycharmm NVE smoke -> ${OUT}"
