#!/usr/bin/env bash
# Full-system PDB → PyCHARMM SD minimize only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/01_from_pdb_pycharmm_minimize.yaml"
OUT="${ARTIFACTS_DIR}/01_mini_pycharmm"

echo "=== config $(basename "${CFG}") (${PDB_MONOMER}) ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-pdb "${PDB_MONOMER}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: pycharmm minimize -> ${OUT}"
