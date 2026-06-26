#!/usr/bin/env bash
# md-system --evaluate-npz (requires PyCHARMM for PSF build).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_cpu/_env.sh"
cd "${ROOT}"

NPZ="${ARTIFACTS_DIR}/aco_dimer.npz"
OUT="${ARTIFACTS_DIR}/evaluate_ase"

echo "=== Build reference geometry NPZ ==="
uv run python examples/md_cpu/02_ml_energy_ase.py \
  --write-npz "${NPZ}" \
  --n-monomers 2

echo "=== md-system --evaluate-npz (ASE, ML-only vacuum) ==="
uv run mmml md-system \
  --evaluate-npz "${NPZ}" \
  --composition ACO:2 \
  --backend ase \
  --setup free_nve \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}" \
  --skip-jit-warmup \
  --quiet

test -f "${OUT}/evaluate.json" || test -f "${OUT}/evaluate_model.extxyz"
echo "PASS: evaluate-npz artifacts in ${OUT}"
