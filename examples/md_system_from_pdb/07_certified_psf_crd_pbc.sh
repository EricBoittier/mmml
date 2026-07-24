#!/usr/bin/env bash
# Certified PSF/CRD box → JAX-MD PBC NVE smoke.
# Set CERTIFIED_BOX_DIR to a liquid-box output (model.psf + model.crd), or
# place mmml_tutorial next to mmml (auto-detected dcm206).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/md_system_from_pdb/_env.sh"
cd "${ROOT}"

CFG="${ROOT}/examples/md_system_from_pdb/yaml/07_certified_psf_crd_pbc.yaml"

if [[ -z "${CERTIFIED_BOX_DIR:-}" ]]; then
  echo "SKIP: no certified box. Build one with:"
  echo "  uv run mmml liquid-box --composition DCM:8 \\"
  echo "    --output-dir artifacts/md_system_from_pdb/box_dcm8"
  echo "  export CERTIFIED_BOX_DIR=artifacts/md_system_from_pdb/box_dcm8"
  echo "  bash examples/md_system_from_pdb/07_certified_psf_crd_pbc.sh"
  echo
  echo "Or edit paths in ${CFG}"
  exit 0
fi

PSF="${CERTIFIED_BOX_DIR}/model.psf"
CRD="${CERTIFIED_BOX_DIR}/model.crd"
if [[ ! -f "${PSF}" || ! -f "${CRD}" ]]; then
  echo "ERROR: missing ${PSF} or ${CRD}" >&2
  exit 1
fi

OUT="${ARTIFACTS_DIR}/07_pbc_nve_jaxmd"

echo "=== config $(basename "${CFG}") (${CERTIFIED_BOX_DIR}) ==="
uv run mmml md-system \
  --config "${CFG}" \
  --from-psf "${PSF}" \
  --from-crd "${CRD}" \
  --checkpoint "${CKPT_JSON}" \
  --output-dir "${OUT}"

echo "PASS: certified PBC NVE -> ${OUT}"
