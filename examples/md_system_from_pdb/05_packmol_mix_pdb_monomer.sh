#!/usr/bin/env bash
# Monomer PDB template → Packmol 4-mer cluster → PyCHARMM free_nve.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/05_packmol_mix_pdb_monomer.yaml"
OUT="${ARTIFACTS_DIR}/05_packmol_4mer"

echo "=== config $(basename "${CFG}") (${PDB_MONOMER}:4) ==="
uv run mmml md-system \
  --config "${CFG}" \
  --composition "${PDB_MONOMER}:4" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: Packmol PDB monomer mix -> ${OUT}"
