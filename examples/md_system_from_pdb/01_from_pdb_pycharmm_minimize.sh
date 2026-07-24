#!/usr/bin/env bash
# Full-system PDB → PyCHARMM SD minimize only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/01_mini_pycharmm"

echo "=== from-pdb → pycharmm_minimize (${PDB_MONOMER}) ==="
uv run mmml md-system \
  --from-pdb "${PDB_MONOMER}" \
  --backend pycharmm \
  --setup pycharmm_minimize \
  --checkpoint "${CKPT_JSON}" \
  --mini-nstep 30 \
  --skip-energy-show \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: pycharmm minimize -> ${OUT}"
