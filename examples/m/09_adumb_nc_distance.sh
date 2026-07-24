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
    # Lone full-system PDB; do not Packmol-rebuild over the NPZ geometry.
    EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}" --no-packmol)
  fi
fi

mkdir -p "${OUT}"
# Drop stale Packmol / pretreat / next_run state from earlier failed attempts
# (a leftover next_run can re-launch a 1000 ps heat instead of the smoke).
rm -rf "${OUT}/.packmol_cache" "${OUT}/packmol_cluster" "${OUT}/pretreat" "${OUT}/cleanup"
rm -f "${OUT}/stage_summary.json" \
  "${OUT}/next_run.yaml" "${OUT}/next_run.sh" "${OUT}/next_run.command"

echo "=== ADUMB N–C distance: $(basename "${CFG}") ==="
echo "     (needs CHARMM ADUMB + ADUMBRXNCOR / ?ADUMBRXN; RXNCOR distance umbrella)"
echo "     If Unknown umbrella / SIGSEGV / 'out of range': rebuild_charmm_mlpot.sh"
echo "     Smoke: ps_heat=0.2 (wipe output_dir if a prior 1000 ps next_run remains)"
echo "     MMML_CGENFF_EXTRA_RTF=${MMML_CGENFF_EXTRA_RTF:-}"
echo "     MMML_CGENFF_EXTRA_PRM=${MMML_CGENFF_EXTRA_PRM:-}"

# Ensure CH3CL append RTF/PRM are visible (sourced from examples/m/_env.sh).
if [[ -z "${MMML_CGENFF_EXTRA_RTF:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_RTF unset — CH3CL will not be in CGenFF"
fi
if [[ -z "${MMML_CGENFF_EXTRA_PRM:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_PRM unset — CG331–CLGA1 bond/angle may be missing"
fi

set +e
uv run mmml md-system \
  --config "${CFG}" \
  --output-dir "${OUT}" \
  "${EXTRA[@]}"
md_rc=$?
set -e
if [[ "${md_rc}" -ne 0 ]]; then
  echo "FAIL: md-system exited ${md_rc}"
  exit "${md_rc}"
fi

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
# Lingo is staged before dynamics; require an ADUMB output so a soft-failed
# md-system (exit 0 + error stages) does not report PASS.
# Library lingo uppercases OPEN names → ADUMB-WUNI.DAT (also accept lowercase).
if [[ ! -f "${OUT}/ADUMB-WUNI.DAT" && ! -f "${OUT}/adumb-wuni.dat" ]]; then
  echo "FAIL: missing ${OUT}/ADUMB-WUNI.DAT (ADUMB did not produce output)"
  exit 1
fi

echo "PASS: ADUMB wiring -> ${OUT}"
echo "      ADUMB-WUNI.DAT / UMBCOR / RXNCOR_TRACE.DAT under ${OUT}"
