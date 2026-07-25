#!/usr/bin/env bash
# PyCHARMM 2D ADUMB on NH3–CH3Cl: Cl⋯C and C⋯N (RXNCOR).
#
# Starting geometry (env vars; choose one):
#   default            build the dimer with Packmol (composition from the YAML)
#   USE_NPZ_PDB=1      seed from an NPZ frame instead: 07_export_solute_pdb.py writes
#                      a centered CGenFF AMM1+CH3CL PDB and md-system runs it via
#                      --from-pdb --no-packmol (the NPZ has no residue/atom names, so
#                      CHARMM cannot read it directly — the PDB carries them).
#     RCL=<a> RCN=<b>    with USE_NPZ_PDB=1: pick the N=9 frame nearest a 2D
#                        (r_ClC, r_CN) target Å — e.g. RCL=3.8 RCN=1.57 = product basin
#     TS_XI=<x>          with USE_NPZ_PDB=1: pick the N=9 frame nearest ξ=r(Cl-C)/r(C-N)
#     FRAME=<n>          with USE_NPZ_PDB=1: pick an absolute N=9 NPZ index
#     SEED_PRESERVE=0    with USE_NPZ_PDB=1: restore the default pre-min (by default the
#                        MM pre-min + monomer mini are skipped so a broken-C-Cl seed survives)
#
# Examples:
#   bash examples/m/10_adumb_clc_cn_2d.sh                                 # Packmol dimer
#   RCL=3.8 RCN=1.57 USE_NPZ_PDB=1 bash examples/m/10_adumb_clc_cn_2d.sh  # seed product basin
#   TS_XI=1.0 USE_NPZ_PDB=1 bash examples/m/10_adumb_clc_cn_2d.sh         # seed near TS
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
  # Seed the starting geometry from an NPZ frame (highest priority first):
  #   RCL + RCN  nearest frame to a 2D (r_ClC, r_CN) target — e.g. RCL=3.8 RCN=1.57
  #              seeds the product basin (broken C-Cl) for the 2D map
  #   TS_XI      nearest frame to a target ξ=r(Cl-C)/r(C-N) (e.g. TS_XI=1.0)
  #   FRAME      absolute N=9 index
  #   (none)     seeded-random frame (--seed 0)
  EXPORT_ARGS=()
  if [[ -n "${RCL:-}" && -n "${RCN:-}" ]]; then
    EXPORT_ARGS+=(--rcl "${RCL}" --rcn "${RCN}")
  elif [[ -n "${TS_XI:-}" ]]; then
    EXPORT_ARGS+=(--xi "${TS_XI}")
  elif [[ -n "${FRAME:-}" ]]; then
    EXPORT_ARGS+=(--frame "${FRAME}")
  fi
  uv run python examples/m/07_export_solute_pdb.py "${EXPORT_ARGS[@]}" -o "${SOLUTE}"
  EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}" --no-packmol)
  # Preserve the seeded reaction coordinate (SEED_PRESERVE=1, default). The NPZ
  # frame is already a physical geometry, so skip the two steps that would erase
  # a broken/dissociated seed before dynamics:
  #   --charmm-sd-steps 0 --charmm-abnr-steps 0  : the full-CGenFF MM pre-min
  #        reforms the C-Cl harmonic bond (BONDs jumps then relaxes to ~1.78 Å);
  #   --no-monomer-physnet-mini                  : isolated-monomer PhysNet mini
  #        pulls CH3Cl back to its gas-phase equilibrium.
  # The hybrid ML BFGS (--calculator-pre-minimize) is kept — it relaxes downhill
  # within the seeded basin only (product stays product; it will not cross the
  # barrier). Set SEED_PRESERVE=0 if a raw seed fails the pre-dynamics GRMS gate.
  if [[ "${SEED_PRESERVE:-1}" == "1" ]]; then
    EXTRA+=(--charmm-sd-steps 0 --charmm-abnr-steps 0 --no-monomer-physnet-mini)
  fi
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
if ! grep -q "name rcl" "${LINGO}" || ! grep -q "name rcn" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing rcl / rcn reaction coordinates"
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
