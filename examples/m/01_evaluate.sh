#!/usr/bin/env bash
# Evaluate kl.json on nh3_ch3cl_filtered.npz (parity metrics + plots).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

OUT="${ARTIFACTS_DIR}/evaluate"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "=== physnet-evaluate (kl.json × nh3_ch3cl_filtered.npz) ==="
uv run mmml physnet-evaluate \
  --checkpoint "${MMML_CKPT}" \
  --data "${MMML_DATA}" \
  --natoms 9 \
  --batch-size "${BATCH_SIZE}" \
  --num-samples "${NUM_SAMPLES}" \
  --plots \
  -o "${OUT}"

test -f "${OUT}/metrics.json"
echo "PASS: evaluate -> ${OUT}"
