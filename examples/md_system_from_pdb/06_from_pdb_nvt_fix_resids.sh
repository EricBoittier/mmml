#!/usr/bin/env bash
# Packmol 4-mer from monomer PDB → free_nvt with constrained SD (fix-resids).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/06_nvt_fix_resids"

echo "=== composition ${PDB_MONOMER}:4 → free_nvt + fix-resids 1,3 ==="
uv run mmml md-system \
  --composition "${PDB_MONOMER}:4" \
  --backend pycharmm \
  --setup free_nvt \
  --checkpoint "${CKPT_JSON}" \
  --packmol-radius 15 \
  --flat-bottom-radius 12 \
  --fix-resids 1,3 \
  --temperature 300 \
  --ps 0.1 \
  --dt-fs 0.5 \
  --mini-nstep 30 \
  --skip-energy-show \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: NVT + fix-resids smoke -> ${OUT}"
echo "Check: fixed monomers should have RMSD ≈ 0 after SD pass 2."
