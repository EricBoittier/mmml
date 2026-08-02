#!/usr/bin/env bash
# Packmol 4-mer from monomer PDB → free_nvt with constrained SD (fix-resids).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/06_from_pdb_nvt_fix_resids.yaml"
OUT="${ARTIFACTS_DIR}/06_nvt_fix_resids"

echo "=== config $(basename "${CFG}") (fix-resids 1,3) ==="
uv run mmml md-system \
  --config "${CFG}" \
  --composition "${PDB_MONOMER}:4" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: NVT + fix-resids smoke -> ${OUT}"
echo "Check: fixed monomers should have RMSD ≈ 0 after SD pass 2."
