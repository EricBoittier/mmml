#!/usr/bin/env bash
# Step 07 — deploy trained LJ scales in condensed-phase MD.
#
# Default (LJ_JOINT!=1): Packmol / seeded DCM campaign (jaxmd-only ladder)
#   jaxmd settle → jaxmd NVT (10 ps) → jaxmd NpT (2 ps) → jaxmd NVE
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
# Default 3.5 eV/Å matches md_fixed_lj_scales_liquid_campaign.yaml (was 2.0).
GATE_ARGS+=(--max-fmax-before-dyn-ev-A "${LJ_MD_MAX_FMAX_EV_A:-3.5}")

# LJ_MD_PROD=1 → tens-of-ps production YAML (RDF / density sampling).
if [[ "${LJ_MD_PROD:-0}" == "1" ]]; then
  DCM_CAMPAIGN_YAML="examples/hybrid_mm_charges/md_fixed_lj_scales_liquid_campaign.prod.yaml"
  JOINT_CAMPAIGN_YAML="examples/hybrid_mm_charges/md_lj_scales_liquid_campaign.prod.yaml"
  echo "  campaign  : PRODUCTION (${DCM_CAMPAIGN_YAML})"
else
  DCM_CAMPAIGN_YAML="examples/hybrid_mm_charges/md_fixed_lj_scales_liquid_campaign.yaml"
  JOINT_CAMPAIGN_YAML="examples/hybrid_mm_charges/md_lj_scales_liquid_campaign.yaml"
fi

if [[ "${LJ_JOINT}" != "1" ]]; then
  echo "=== 07: DCM liquid campaign (jaxmd settle → NVT 10 ps → NpT 2 ps) ==="
  # Knobs:
  #   LJ_MD_PROD=1                  20 ps NVT (+ same 2 ps NpT probe)
  #   LJ_MD_PACKMOL_TOLERANCE=3.5   pack further apart (Packmol path only)
  #   LJ_MD_MAX_FMAX_EV_A=3.5       pre-dynamics force gate (eV/Å)
  # Optional seed (skips Packmol rebuild when mini.crd already exists):
  #   LJ_MD_FROM_PSF / LJ_MD_FROM_CRD, or default liquid_nvt/mini.{psf,crd}
  # Fresh campaign dir each run unless you pass --resume (via LJ_MD_RESUME=1).
  # After a failed CHARMM heat from an older campaign, wipe liquid_dcm and
  # re-run — do not resume from a blown heat.0.res.
  SEED_PSF="${LJ_MD_FROM_PSF:-${LJ_ARTIFACTS_DIR}/liquid_nvt/mini.psf}"
  SEED_CRD="${LJ_MD_FROM_CRD:-${LJ_ARTIFACTS_DIR}/liquid_nvt/mini.crd}"
  SEED_ARGS=()
  if [[ -f "${SEED_PSF}" && -f "${SEED_CRD}" ]]; then
    echo "  seed box  : ${SEED_PSF} + ${SEED_CRD}"
    SEED_ARGS+=(--from-psf "${SEED_PSF}" --from-crd "${SEED_CRD}" --no-packmol --box-size 30.0)
  fi
  RESUME_ARGS=()
  if [[ "${LJ_MD_RESUME:-0}" == "1" ]]; then
    RESUME_ARGS+=(--resume)
  fi
  OUT_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm"
  if [[ "${LJ_MD_PROD:-0}" == "1" ]]; then
    OUT_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm_prod"
  fi
  uv run mmml md-system \
    --config "${DCM_CAMPAIGN_YAML}" \
    --run-all \
    --checkpoint "${CKPT}" \
    --mm-lj-scales-file "${SIDECAR}" \
    --campaign-output-dir "${OUT_DIR}" \
    --mm-nonbond-mode jax_mic \
    ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
    ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
  echo "07: OK  ${OUT_DIR}"
  echo "     next: bash examples/lj_scales/12_analyze_liquid.sh"
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
    --config "${JOINT_CAMPAIGN_YAML}" \
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

DCM_OUT="${LJ_ARTIFACTS_DIR}/liquid_dcm"
ACO_OUT="${LJ_ARTIFACTS_DIR}/liquid_aco"
if [[ "${LJ_MD_PROD:-0}" == "1" ]]; then
  DCM_OUT="${LJ_ARTIFACTS_DIR}/liquid_dcm_prod"
  ACO_OUT="${LJ_ARTIFACTS_DIR}/liquid_aco_prod"
fi
_run_solvent DCM "${LJ_BOX_DCM_DIR}" "${DCM_OUT}"
_run_solvent ACO "${LJ_BOX_ACO_DIR}" "${ACO_OUT}"

echo "07: OK  ${DCM_OUT}  ${ACO_OUT}"
echo "     next: bash examples/lj_scales/12_analyze_liquid.sh"
