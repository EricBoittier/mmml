#!/usr/bin/env bash
# Step 05 — train ML weights + learnable per-type LJ scales.
#
# lr_solver MUST stay 'mic'. Under ewald / nvalchemiops_pme the LJ term is
# removed from the hybrid energy entirely, so learn_mm_lj_scales has nothing to
# differentiate and is silently ineffective. This is deliberate, not a gap —
# see docs/hybrid-mm-lj-scales.md "Why you train LJ without Ewald".
#
# This is the expensive step. Prefer a GPU:
#     LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh
# Cheap first pass to check the scales move at all:
#     LJ_EPOCHS=50 LJ_NTRAIN=2000 bash examples/lj_scales/05_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 05: physnet-train (learn_mm_lj_scales) ==="

if [[ ! -f "${LJ_ENRICHED}" ]]; then
  echo "ERROR: ${LJ_ENRICHED} not found — run 01_prepare_dataset.sh first." >&2
  exit 2
fi

echo "  epochs=${LJ_EPOCHS} n_train=${LJ_NTRAIN} n_valid=${LJ_NVALID}"
echo "  ckpt_dir=${LJ_CKPT_DIR} tag=${LJ_TAG}"

uv run mmml physnet-train \
  --config examples/hybrid_mm_charges/train_fixed_lj_scales.yaml \
  --data "${LJ_ENRICHED}" \
  --valid-data "" \
  --ckpt-dir "${LJ_CKPT_DIR}" \
  --tag "${LJ_TAG}" \
  --n-train "${LJ_NTRAIN}" \
  --n-valid "${LJ_NVALID}" \
  --num-epochs "${LJ_EPOCHS}"

# Newest, not first: earlier runs leave their own hybrid_mm.json behind, and
# reporting one of those would claim success for a run that never finished.
SIDECAR="$(lj_newest_file "${LJ_CKPT_DIR}" -name hybrid_mm.json || true)"
if [[ -z "${SIDECAR}" ]]; then
  echo "ERROR: training wrote no hybrid_mm.json under ${LJ_CKPT_DIR}" >&2
  echo "       learn_mm_lj_scales was probably forced off. Check the log for" >&2
  echo "       'Learnable MM LJ scales enabled' and confirm lr_solver=mic." >&2
  exit 3
fi
echo "05: OK  ${SIDECAR}"
