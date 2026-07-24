#!/usr/bin/env bash
# Full-system PDB → ASE vacuum NVE smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/02_free_nve_ase"

echo "=== from-pdb → free_nve (ASE, 0.1 ps) ==="
uv run mmml md-system \
  --from-pdb "${PDB_MONOMER}" \
  --backend ase \
  --setup free_nve \
  --checkpoint "${CKPT_JSON}" \
  --ps 0.1 \
  --dt-fs 0.5 \
  --skip-jit-warmup \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: ASE NVE smoke -> ${OUT}"
