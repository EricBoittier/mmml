#!/usr/bin/env bash
# Step 07 — deploy the trained scales in a condensed-phase MD run.
#
# jax_mic is mandatory. ep_scale/sig_scale are consumed by the JAX switched-MM
# pair loop, which is active only when `do_mm = include_mm and not periodic_mode`.
# Under periodic_external, VDW is handed to CHARMM IMAGE, which does not read
# hybrid_mm.json — MLpot now raises rather than silently ignoring the sidecar.
#
# Consequence: this run uses truncated-MIC electrostatics, not Ewald. Combining
# learned LJ with full Ewald is not implemented (issue #139).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 07: md-system liquid_nvt (jax_mic) ==="

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable — cannot run the MD leg." >&2
  exit 0
fi

# Newest match, not first: several runs share LJ_CKPT_DIR and find's traversal
# order is arbitrary. `|| true` because the dir may not exist yet, and under
# `set -e` that would abort before the explanatory error below can print.
#
# Training writes the portable checkpoint as `params_<tag>_<timestamp>.json`
# (make_training.py), so matching only `params.json` finds nothing.
CKPT="${LJ_MD_CKPT:-$(lj_newest_file "${LJ_CKPT_DIR}" \
  \( -name 'params_*.json' -o -name 'params.json' \) || true)}"
SIDECAR="${LJ_SIDECAR:-$(lj_newest_file "${LJ_CKPT_DIR}" -name hybrid_mm.json || true)}"

if [[ -z "${CKPT}" || -z "${SIDECAR}" ]]; then
  echo "ERROR: need both a checkpoint and hybrid_mm.json under ${LJ_CKPT_DIR}" >&2
  echo "       ckpt='${CKPT}' sidecar='${SIDECAR}' — run 05_train.sh first." >&2
  exit 2
fi

echo "  checkpoint : ${CKPT}"
echo "  scales     : ${SIDECAR}"

uv run mmml md-system \
  --config examples/hybrid_mm_charges/md_fixed_lj_scales.yaml \
  --only liquid_nvt \
  --checkpoint "${CKPT}" \
  --mm-lj-scales-file "${SIDECAR}" \
  --output-dir "${ARTIFACTS_DIR}/liquid_nvt"

echo "07: OK  ${ARTIFACTS_DIR}/liquid_nvt"
