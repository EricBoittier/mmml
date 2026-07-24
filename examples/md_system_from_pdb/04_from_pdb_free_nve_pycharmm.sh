#!/usr/bin/env bash
# Full-system PDB → PyCHARMM vacuum NVE with flat-bottom restraint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/04_free_nve_pycharmm"

echo "=== from-pdb → free_nve (pycharmm, 0.1 ps) ==="
uv run mmml md-system \
  --from-pdb "${PDB_MONOMER}" \
  --backend pycharmm \
  --setup free_nve \
  --checkpoint "${CKPT_JSON}" \
  --flat-bottom-radius 20 \
  --ps 0.1 \
  --dt-fs 0.5 \
  --mini-nstep 20 \
  --skip-energy-show \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: pycharmm NVE smoke -> ${OUT}"
