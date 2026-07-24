#!/usr/bin/env bash
# PyCHARMM 2D ADUMB on NH3–CH3Cl: Cl⋯C and C⋯N (RXNCOR).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/adumb_clc_cn_2d.yaml}"
OUT="${ARTIFACTS_DIR}/adumb_clc_cn_2d"
USE_NPZ_PDB="${USE_NPZ_PDB:-0}"

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable"
  exit 0
fi

EXTRA=()
if [[ "${USE_NPZ_PDB}" == "1" ]]; then
  SOLUTE="${ARTIFACTS_DIR}/solute_amm1_ch3cl.pdb"
  uv run python examples/m/07_export_solute_pdb.py -o "${SOLUTE}"
  EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}" --no-packmol)
fi

mkdir -p "${OUT}"
rm -rf "${OUT}/.packmol_cache" "${OUT}/packmol_cluster" "${OUT}/pretreat" "${OUT}/cleanup"
rm -f "${OUT}/stage_summary.json" \
  "${OUT}/next_run.yaml" "${OUT}/next_run.sh" "${OUT}/next_run.command" \
  "${OUT}/pycharmm_pre_dynamics_lingo.inp"

echo "=== ADUMB 2D Cl–C / C–N: $(basename "${CFG}") ==="
echo "     (needs CHARMM ADUMB + ADUMBRXNCOR; two umbrella rxncor cards)"
echo "     MMML_CGENFF_EXTRA_RTF=${MMML_CGENFF_EXTRA_RTF:-}"
echo "     MMML_CGENFF_EXTRA_PRM=${MMML_CGENFF_EXTRA_PRM:-}"

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
if ! grep -q "r_cl" "${LINGO}" || ! grep -q "r_cn" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing r_cl / r_cn reaction coordinates"
  exit 1
fi
n_umb="$(grep -cE '^[[:space:]]*umbrella[[:space:]]+rxncor' "${LINGO}" || true)"
if [[ "${n_umb}" -lt 2 ]]; then
  echo "FAIL: expected ≥2 'umbrella rxncor' cards, found ${n_umb}"
  exit 1
fi
if ! grep -q "nrxn 2" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing 'rxncor set nrxn 2'"
  exit 1
fi
# Unpatched UM1RXN: min>0 shrinks the upper edge to (max-min). Prefer min 0.0.
if grep -E 'umbrella[[:space:]]+rxncor' "${LINGO}" | grep -Eq 'min[[:space:]]+[1-9]'; then
  echo "FAIL: ${LINGO} has umbrella rxncor min>0 — use min 0.0 (or rebuild with UM1RXN patch)"
  exit 1
fi
if [[ ! -f "${OUT}/ADUMB-WUNI.DAT" && ! -f "${OUT}/adumb-wuni.dat" ]]; then
  echo "FAIL: missing ${OUT}/ADUMB-WUNI.DAT (ADUMB did not produce output)"
  exit 1
fi

echo "PASS: ADUMB 2D wiring -> ${OUT}"
echo "      ADUMB-WUNI.DAT / UMBCOR / RXNCOR_RCL.DAT / RXNCOR_RCN.DAT under ${OUT}"
