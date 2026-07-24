#!/usr/bin/env bash
# Monomer PDB template → Packmol 4-mer cluster → PyCHARMM free_nve.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/05_packmol_4mer"

echo "=== composition ${PDB_MONOMER}:4 → free_nve (pycharmm) ==="
uv run mmml md-system \
  --composition "${PDB_MONOMER}:4" \
  --backend pycharmm \
  --setup free_nve \
  --checkpoint "${CKPT_JSON}" \
  --packmol-radius 15 \
  --flat-bottom-radius 12 \
  --ps 0.1 \
  --dt-fs 0.5 \
  --mini-nstep 20 \
  --fix-resids 1 \
  --skip-energy-show \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: Packmol PDB monomer mix -> ${OUT}"
