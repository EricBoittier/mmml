#!/usr/bin/env bash
# Step 11 — certify pure DCM and pure ACO liquid boxes (MM-only).
#
# Uses mmml liquid-box at experimental bulk densities (or a fraction for smoke).
# Hybrid MD (step 07 campaign) loads the certified PSF/CRD afterward.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 11: liquid-box (pure DCM + pure ACO) ==="

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable — cannot certify liquid boxes." >&2
  exit 0
fi

L="${LJ_BOX_SIZE}"
FRAC="${LJ_BULK_DENSITY_FRACTION}"

_build_one() {
  local resid="$1" out="$2"
  echo "--- ${resid}  L=${L} Å  bulk_frac=${FRAC} -> ${out}"
  mkdir -p "${out}"
  # --box-auto count sizes N from L and experimental bulk × fraction.
  # Do not also pass full --target-density-g-cm3 (that would fight the fraction).
  uv run mmml liquid-box \
    --composition "${resid}:1" \
    --box-auto count \
    --box-size "${L}" \
    --bulk-density-fraction "${FRAC}" \
    --output-dir "${out}"
  if [[ ! -f "${out}/model.psf" || ! -f "${out}/model.crd" ]]; then
    echo "ERROR: ${out} missing model.psf / model.crd" >&2
    exit 3
  fi
  if [[ -f "${out}/box.json" ]]; then
    uv run python -c "
import json,sys
b=json.load(open(sys.argv[1]))
print(f\"  status={b.get('status')}  N={b.get('n_molecules')}  \"
      f\"L={b.get('box_side_A')}  rho={b.get('density_g_cm3')}\")
" "${out}/box.json"
  fi
}

_build_one DCM "${LJ_BOX_DCM_DIR}"
_build_one ACO "${LJ_BOX_ACO_DIR}"

echo "11: OK  ${LJ_BOX_DCM_DIR}  ${LJ_BOX_ACO_DIR}"
