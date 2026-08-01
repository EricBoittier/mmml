#!/usr/bin/env bash
# Step 07 — deploy trained LJ scales in condensed-phase MD.
#
# Default (LJ_JOINT!=1): Packmol DCM campaign
#   jaxmd settle → PyCHARMM NVT heat/eq → jaxmd NVT → jaxmd NVE
#
# Joint path (LJ_JOINT=1): certified boxes from step 11 → campaign
#   jaxmd settle → PyCHARMM NpT heat/eq → jaxmd NVT → jaxmd NVE
# for pure DCM and pure ACO. jax_mic is mandatory (learned LJ + Ewald = #139).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable — cannot run the MD leg." >&2
  exit 0
fi

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

GATE_ARGS=()
if [[ -n "${LJ_MD_PACKMOL_TOLERANCE:-}" ]]; then
  GATE_ARGS+=(--packmol-tolerance "${LJ_MD_PACKMOL_TOLERANCE}")
fi
if [[ -n "${LJ_MD_MAX_FMAX_EV_A:-}" ]]; then
  GATE_ARGS+=(--max-fmax-before-dyn-ev-A "${LJ_MD_MAX_FMAX_EV_A}")
fi

if [[ "${LJ_JOINT}" != "1" ]]; then
  echo "=== 07: DCM liquid campaign (jaxmd settle → PyCHARMM NVT → jaxmd) ==="
  # Pre-dynamics force gate knobs (Packmol path only):
  #   LJ_MD_PACKMOL_TOLERANCE=3.5   pack further apart (try this first)
  #   LJ_MD_MAX_FMAX_EV_A=3.0       raise the ceiling after inspecting the frame
  # Tolerance is part of the packmol cache key; changing it rebuilds the box.
  uv run mmml md-system \
    --config examples/hybrid_mm_charges/md_fixed_lj_scales_liquid_campaign.yaml \
    --run-all \
    --checkpoint "${CKPT}" \
    --mm-lj-scales-file "${SIDECAR}" \
    --campaign-output-dir "${LJ_ARTIFACTS_DIR}/liquid_dcm" \
    --mm-nonbond-mode jax_mic \
    ${GATE_ARGS[@]+"${GATE_ARGS[@]}"}
  echo "07: OK  ${LJ_ARTIFACTS_DIR}/liquid_dcm"
  exit 0
fi

echo "=== 07: joint liquid campaign (jaxmd → PyCHARMM NpT → jaxmd NVT/NVE) ==="

_run_solvent() {
  local resid="$1" box_dir="$2" out_root="$3"
  local psf="${box_dir}/model.psf"
  local crd="${box_dir}/model.crd"
  if [[ ! -f "${psf}" || ! -f "${crd}" ]]; then
    echo "ERROR: certified box missing under ${box_dir}" >&2
    echo "       Run 11_liquid_boxes.sh first." >&2
    exit 2
  fi
  # Read N from box.json when present so composition matches the certified box.
  local comp
  comp="$(uv run python - "${box_dir}" "${resid}" <<'PY'
import json, sys
from pathlib import Path
box = Path(sys.argv[1])
resid = sys.argv[2]
n = None
bj = box / "box.json"
if bj.is_file():
    b = json.loads(bj.read_text())
    n = b.get("n_molecules")
    comp = b.get("composition")
    if comp:
        print(comp)
        raise SystemExit(0)
if n is None:
    # Fallback: count RESI lines is fragile; use resid:1 and let md-system
    # inherit geometry from PSF/CRD (composition still required by CLI).
    print(f"{resid}:1")
else:
    print(f"{resid}:{int(n)}")
PY
)"
  echo "--- ${resid}: composition=${comp}  box=${box_dir}  out=${out_root}"
  mkdir -p "${out_root}"
  uv run mmml md-system \
    --config examples/hybrid_mm_charges/md_lj_scales_liquid_campaign.yaml \
    --run-all \
    --checkpoint "${CKPT}" \
    --mm-lj-scales-file "${SIDECAR}" \
    --from-psf "${psf}" \
    --from-crd "${crd}" \
    --composition "${comp}" \
    --campaign-output-dir "${out_root}" \
    --mm-nonbond-mode jax_mic \
    ${GATE_ARGS[@]+"${GATE_ARGS[@]}"}
}

_run_solvent DCM "${LJ_BOX_DCM_DIR}" "${LJ_ARTIFACTS_DIR}/liquid_dcm"
_run_solvent ACO "${LJ_BOX_ACO_DIR}" "${LJ_ARTIFACTS_DIR}/liquid_aco"

echo "07: OK  ${LJ_ARTIFACTS_DIR}/liquid_dcm  ${LJ_ARTIFACTS_DIR}/liquid_aco"
