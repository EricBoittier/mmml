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

# Pre-dynamics force gate overrides. The gate rejects a starting frame whose
# worst single-atom force exceeds 2 eV/Å. Every molecule here is ML, so CHARMM
# GRMS is 0 and CHARMM SD cannot help; only looser packing or a raised ceiling
# gets past it. Tolerance is part of the packmol cache key, so changing it
# rebuilds the box without --rebuild-packmol.
#
#   LJ_MD_PACKMOL_TOLERANCE=3.5   pack further apart (try this first)
#   LJ_MD_MAX_FMAX_EV_A=3.0       raise the ceiling once you have inspected the frame
GATE_ARGS=()
if [[ -n "${LJ_MD_PACKMOL_TOLERANCE:-}" ]]; then
  GATE_ARGS+=(--packmol-tolerance "${LJ_MD_PACKMOL_TOLERANCE}")
fi
if [[ -n "${LJ_MD_MAX_FMAX_EV_A:-}" ]]; then
  GATE_ARGS+=(--max-fmax-before-dyn-ev-A "${LJ_MD_MAX_FMAX_EV_A}")
fi

# --job-id, not --only: md-system has no --only flag (that one belongs to the
# ase/jaxmd pbc suite it shells out to) and campaign dispatch keys off --job-id.
# --output-dir overrides the job's YAML output_dir only because --job-id pins the
# campaign to a single run; with --run-all the CLI rejects it as ambiguous.
uv run mmml md-system \
  --config examples/hybrid_mm_charges/md_fixed_lj_scales.yaml \
  --job-id liquid_nvt \
  --checkpoint "${CKPT}" \
  --mm-lj-scales-file "${SIDECAR}" \
  --output-dir "${LJ_ARTIFACTS_DIR}/liquid_nvt" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"}

echo "07: OK  ${LJ_ARTIFACTS_DIR}/liquid_nvt"
