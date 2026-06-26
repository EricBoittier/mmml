#!/usr/bin/env bash
# Full md-system vacuum NVE via ASE backend (requires PyCHARMM for cluster PSF).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_cpu/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/free_nve_ase"

echo "=== md-system free_nve (ASE backend, 0.1 ps) ==="
uv run mmml md-system \
  --setup free_nve \
  --backend ase \
  --composition ACO:2 \
  --spacing 5.0 \
  --checkpoint "${CKPT_JSON}" \
  --ps 0.1 \
  --dt-fs 0.5 \
  --skip-jit-warmup \
  --output-dir "${OUT}" \
  --quiet

echo "PASS: ASE NVE smoke -> ${OUT}"
