#!/usr/bin/env bash
# PyCHARMM ADUMB on NH3–CH3Cl N⋯C distance (from examples/m NPZ / Packmol).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/adumb_nc_distance.yaml}"
OUT="${ARTIFACTS_DIR}/adumb_nc_distance"
USE_NPZ_PDB="${USE_NPZ_PDB:-0}"
SOLVATED="${SOLVATED:-0}"

if [[ "${SOLVATED}" == "1" ]]; then
  CFG="${ROOT}/examples/m/yaml/adumb_nc_distance_tip3.yaml"
  OUT="${ARTIFACTS_DIR}/adumb_nc_distance_tip3"
fi

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable"
  exit 0
fi

# Optional: feed coordinates from the NPZ-exported CGenFF PDB.
# YAML still has Packmol composition (AMM1:1,CH3CL:1[…]); --from-pdb alone
# cannot mix with that — override --composition to a lone PDB (vacuum) or
# solute.pdb:1,TIP3:N (solvated).
EXTRA=()
if [[ "${USE_NPZ_PDB}" == "1" ]]; then
  SOLUTE="${ARTIFACTS_DIR}/solute_amm1_ch3cl.pdb"
  uv run python examples/m/07_export_solute_pdb.py -o "${SOLUTE}"
  if [[ "${SOLVATED}" == "1" ]]; then
    EXTRA+=(--composition "${SOLUTE}:1,TIP3:12")
  else
    EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}")
  fi
fi

mkdir -p "${OUT}"
echo "=== ADUMB N–C distance: $(basename "${CFG}") ==="
echo "     (needs CHARMM ADUMB + ADUMBRXN; RXNCOR distance umbrella)"
echo "     MMML_CGENFF_EXTRA_RTF=${MMML_CGENFF_EXTRA_RTF:-}"

# Ensure CH3CL append RTF is visible (sourced from examples/m/_env.sh).
if [[ -z "${MMML_CGENFF_EXTRA_RTF:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_RTF unset — CH3CL will not be in CGenFF"
fi

uv run mmml md-system \
  --config "${CFG}" \
  --output-dir "${OUT}" \
  "${EXTRA[@]}"

LINGO="${OUT}/pycharmm_pre_dynamics_lingo.inp"
if [[ ! -f "${LINGO}" ]]; then
  echo "FAIL: missing ${LINGO}"
  exit 1
fi
if ! grep -q "umbrella rxncor" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing 'umbrella rxncor'"
  exit 1
fi
if ! grep -q "r_nc" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing r_nc reaction coordinate"
  exit 1
fi

echo "PASS: ADUMB wiring -> ${OUT}"
echo "      Check adumb-wuni.dat / umbcor / rxncor_trace.dat under ${OUT}"
