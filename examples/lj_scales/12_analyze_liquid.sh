#!/usr/bin/env bash
# Step 12 — density / RDF / MSD / timeseries plots from liquid campaign HDF5.
#
# Prefers production dirs (liquid_*_prod) when present, else smoke liquid_*.
# Does not run MD. Safe to re-run after each campaign.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 12: analyze liquid campaign trajectories ==="

_analyze_one() {
  local label="$1"
  local campaign_dir="$2"
  local solvent="$3"
  if [[ ! -d "${campaign_dir}" ]]; then
    echo "SKIP ${label}: missing ${campaign_dir}"
    return 0
  fi
  local out="${campaign_dir}/analysis"
  echo "--- ${label}: ${campaign_dir} → ${out}"
  uv run mmml analyze-liquid \
    --campaign-dir "${campaign_dir}" \
    --solvent "${solvent}" \
    --prefer-run "${LJ_ANALYZE_PREFER_RUN:-jaxmd_npt}" \
    --max-frames "${LJ_ANALYZE_MAX_FRAMES:-400}" \
    -o "${out}"
}

if [[ "${LJ_JOINT}" == "1" ]]; then
  DCM_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm_prod"
  [[ -d "${DCM_DIR}" ]] || DCM_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm"
  ACO_DIR="${LJ_ARTIFACTS_DIR}/liquid_aco_prod"
  [[ -d "${ACO_DIR}" ]] || ACO_DIR="${LJ_ARTIFACTS_DIR}/liquid_aco"
  _analyze_one DCM "${DCM_DIR}" DCM
  _analyze_one ACO "${ACO_DIR}" ACO
else
  DCM_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm_prod"
  [[ -d "${DCM_DIR}" ]] || DCM_DIR="${LJ_ARTIFACTS_DIR}/liquid_dcm"
  # Also accept UUID-suffixed smoke dirs created when liquid_dcm already existed.
  if [[ ! -d "${DCM_DIR}" ]]; then
    newest="$(ls -dt "${LJ_ARTIFACTS_DIR}"/liquid_dcm* 2>/dev/null | head -1 || true)"
    if [[ -n "${newest}" ]]; then
      DCM_DIR="${newest}"
    fi
  fi
  _analyze_one DCM "${DCM_DIR}" DCM
fi

echo "12: OK — see */analysis/metrics.json and PNGs"
echo "    Pass criteria: RDF first peaks finite; for NpT prod, |Δρ| vs bulk ≲ 5–10%."
