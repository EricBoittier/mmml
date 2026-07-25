#!/usr/bin/env bash
# PyCHARMM ADUMB on NH3–CH3Cl bond ratio ξ=r(Cl-C)/r(C-N).
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
#     SEED_PRESERVE=0    with USE_NPZ_PDB=1 (vacuum): restore the default pre-min (by default
#                        MM pre-min + monomer mini are skipped so a broken-C-Cl seed survives)
#   SOLVATED=1         use the explicit-TIP3 (PBC) YAML instead of vacuum
#
# Examples:
#   bash examples/m/09_adumb_nc_distance.sh                            # Packmol dimer
#   TS_XI=1.0 USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh    # seed near TS
#   FRAME=4101 USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh   # exact frame
# Many seeds across ξ (independent full-range replicas): examples/m/11_adumb_windows.sh
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
  # Seed the starting geometry from an NPZ frame (highest priority first):
  #   RCL + RCN  nearest frame to a 2D (r_ClC, r_CN) target — e.g. RCL=3.8 RCN=1.57
  #              seeds the product basin (broken C-Cl)
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
  if [[ "${SOLVATED}" == "1" ]]; then
    # Solvated: Packmol wraps TIP3 around the solute — keep the default pre-min.
    EXTRA+=(--composition "${SOLUTE}:1,TIP3:12")
  else
    # Lone full-system PDB; do not Packmol-rebuild over the NPZ geometry.
    EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}" --no-packmol)
    # Preserve the seeded reaction coordinate (SEED_PRESERVE=1, default): skip the
    # full-CGenFF MM pre-min (reforms the C-Cl harmonic bond) and the isolated-
    # monomer PhysNet mini (pulls CH3Cl back to gas-phase equilibrium), both of
    # which would erase a broken/dissociated seed. The hybrid ML BFGS
    # (--calculator-pre-minimize) is kept: it only relaxes within the seeded basin.
    # SEED_PRESERVE=0 restores the default pre-min if a raw seed fails the GRMS gate.
    if [[ "${SEED_PRESERVE:-1}" == "1" ]]; then
      EXTRA+=(--charmm-sd-steps 0 --charmm-abnr-steps 0 --no-monomer-physnet-mini)
    fi
  fi
fi

mkdir -p "${OUT}"
# Drop stale Packmol / pretreat / next_run / lingo state from earlier attempts.
# A leftover pycharmm_pre_dynamics_lingo.inp can keep old umbrella min/max
# (e.g. min 2 max 6 → UM1RXN "out of range") after YAML edits.
rm -rf "${OUT}/.packmol_cache" "${OUT}/packmol_cluster" "${OUT}/pretreat" "${OUT}/cleanup"
rm -f "${OUT}/stage_summary.json" \
  "${OUT}/next_run.yaml" "${OUT}/next_run.sh" "${OUT}/next_run.command" \
  "${OUT}/pycharmm_pre_dynamics_lingo.inp"

echo "=== ADUMB Cl–C / C–N ratio: $(basename "${CFG}") ==="
echo "     (needs CHARMM ADUMB + ADUMBRXNCOR / ?ADUMBRXN; RXNCOR ratio umbrella)"
echo "     If Unknown umbrella / SIGSEGV / 'out of range': rebuild_charmm_mlpot.sh"
echo "     Default YAML: ps_heat=100, ξ in [0.125, 5.0] (needs UM1RXN patch for min>0)"
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
if ! grep -Eqi 'ratio[[:space:]]+rcl[[:space:]]+rcn|define[[:space:]]+rrat[[:space:]]+ratio' "${LINGO}"; then
  echo "FAIL: ${LINGO} missing rrat = ratio rcl rcn"
  exit 1
fi
if ! grep -q "rrat" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing rrat reaction coordinate"
  exit 1
fi
# Stale distance-only lingo after switching to ratio.
if grep -Eq 'name[[:space:]]+r_nc|define[[:space:]]+r_nc[[:space:]]+distance' "${LINGO}"; then
  echo "FAIL: ${LINGO} still has old r_nc distance RC — wipe output_dir and re-run"
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
