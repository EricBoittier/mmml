#!/usr/bin/env bash
# Full-system PDB → JAX-MD vacuum NVE smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/03_free_nve_jaxmd"

echo "=== from-pdb → free_nve (jaxmd, 0.1 ps) ==="
uv run mmml md-system \
  --from-pdb "${PDB_MONOMER}" \
  --backend jaxmd \
  --setup free_nve \
  --checkpoint "${CKPT_JSON}" \
  --ps 0.1 \
  --dt-fs 0.5 \
  --skip-jit-warmup \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: jaxmd NVE smoke -> ${OUT}"
