#!/usr/bin/env bash
# PyCHARMM ADUMB on NH3–CH3Cl bond difference ξ=r(Cl-C)−r(C-N) ∈ [-3, 3] Å.
#
# Starting geometry (env vars; choose one):
#   default            build the dimer with Packmol (composition from the YAML)
#   USE_NPZ_PDB=1      seed from an NPZ frame instead: 07_export_solute_pdb.py writes
#                      a centered CGenFF AMM1+CH3CL PDB and md-system runs it via
#                      --from-pdb --no-packmol (the NPZ has no residue/atom names, so
#                      CHARMM cannot read it directly — the PDB carries them).
#     RCL=<a> RCN=<b>    with USE_NPZ_PDB=1: pick the N=9 frame nearest a 2D
#                        (r_ClC, r_CN) target Å — e.g. RCL=3.8 RCN=1.57 = product basin
#     TS_XI=<x>          with USE_NPZ_PDB=1: pick nearest frame to ratio ξ (export helper)
#     FRAME=<n>          with USE_NPZ_PDB=1: pick an absolute N=9 NPZ index
#     SEED_PRESERVE=0    with USE_NPZ_PDB=1 (vacuum or solvated): restore the default
#                        pre-min (by default MM pre-min + monomer mini are skipped so a
#                        broken-C-Cl seed survives)
#   SOLVATED=1           Packmol from YAML (AMM1:1,CH3CL:1,SOLVENT:N)
#   SOLVATED=1 USE_NPZ_PDB=1
#                        export solute → rebuild make-box for SOLVENT → --from-pdb box
#                        (cannot Packmol a multi-residue solute PDB as one monomer)
#
# Requires mmml-patched libcharmm (UM1RXN [min,max]). Do NOT leave CHARMM_LIB_DIR
# pointing at a stale PhysNet_PyCHARMM tree without that patch.
#
# Examples:
#   bash examples/m/09_adumb_nc_distance.sh                            # Packmol dimer
#   FRAME=1000 USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh  # exact frame
#   SOLVATED=1 SOLVENT=dmso USE_NPZ_PDB=1 bash examples/m/09_adumb_nc_distance.sh
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
SOLVENT="${SOLVENT:-tip3}"

if [[ "${SOLVATED}" == "1" ]]; then
  CFG="${ROOT}/examples/m/yaml/adumb_nc_distance_${SOLVENT}.yaml"
  OUT="${ARTIFACTS_DIR}/adumb_nc_distance_${SOLVENT}"
fi

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable"
  exit 0
fi

# Warn/fail if CHARMM_LIB_DIR looks like an old external PhysNet build (no UM1RXN patch).
if [[ -n "${CHARMM_LIB_DIR:-}" && "${CHARMM_LIB_DIR}" == *"PhysNet_PyCHARMM"* ]]; then
  echo "FAIL: CHARMM_LIB_DIR=${CHARMM_LIB_DIR}"
  echo "      That tree is unpatched (UM1RXN). Rebuild mmml CHARMM and retarget:"
  echo "        bash scripts/rebuild_charmm_mlpot.sh --clean"
  echo "        export CHARMM_LIB_DIR=${ROOT}/setup/charmm/lib"
  exit 1
fi

# Optional: feed coordinates from the NPZ-exported CGenFF PDB.
# YAML still has Packmol composition (AMM1:1,CH3CL:1[…]); override it with a
# lone full-system PDB (vacuum) or a make-box solvent PDB (solvated).
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
    # Solvated: rebuild make-box around the seeded solute, then cold-start from
    # that full-system PDB. Packmol cannot take AMM1+CH3CL as one monomer
    # (``solute.pdb:1,TIP3:N`` → "single residue name" error).
    BOX_PDB="${ARTIFACTS_DIR}/boxes/${SOLVENT}/model.pdb"
    echo "=== rebuild make-box ${SOLVENT} around NPZ solute ==="
    SOLUTE_PDB="${SOLUTE}" \
      BOX_SIZE="${BOX_SIZE:-30.0}" \
      N_SOLVENT="${N_SOLVENT:-12}" \
      SOLVENT_ONLY="${SOLVENT}" \
      SKIP_SOLUTE_EXPORT=1 \
      bash examples/m/08_make_boxes.sh
    if [[ ! -f "${BOX_PDB}" ]]; then
      echo "FAIL: missing ${BOX_PDB} after make-box" >&2
      exit 1
    fi
    EXTRA+=(--composition "${BOX_PDB}" --from-pdb "${BOX_PDB}" --no-packmol)
  else
    # Lone full-system PDB; do not Packmol-rebuild over the NPZ geometry.
    EXTRA+=(--composition "${SOLUTE}" --from-pdb "${SOLUTE}" --no-packmol)
  fi
  # Preserve the seeded reaction coordinate (SEED_PRESERVE=1, default): skip the
  # full-CGenFF MM pre-min (reforms the C-Cl harmonic bond) and the isolated-
  # monomer PhysNet mini (pulls CH3Cl back to gas-phase equilibrium), both of
  # which would erase a broken/dissociated seed. Applies to vacuum and solvated
  # USE_NPZ_PDB paths. The hybrid ML BFGS (--calculator-pre-minimize) is kept.
  # SEED_PRESERVE=0 restores the default pre-min if a raw seed fails the GRMS gate.
  if [[ "${SEED_PRESERVE:-1}" == "1" ]]; then
    EXTRA+=(--charmm-sd-steps 0 --charmm-abnr-steps 0 --no-monomer-physnet-mini)
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

echo "=== ADUMB Cl–C − C–N difference: $(basename "${CFG}") ==="
echo "     (needs CHARMM ADUMB + ADUMBRXNCOR + UM1RXN [min,max] patch)"
echo "     If Unknown umbrella / SIGSEGV / 'out of range': rebuild_charmm_mlpot.sh"
echo "     Default YAML: ps_heat=100, ξ=r(ClC)−r(CN) ∈ [-6, 6] Å (SN2 band ~[-3,3])"
echo "     CHARMM_LIB_DIR=${CHARMM_LIB_DIR:-<unset>}"
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
if ! grep -Eqi 'combination[[:space:]]+rcl|[[:space:]]+rdif[[:space:]]+combination' "${LINGO}"; then
  echo "FAIL: ${LINGO} missing rdif = combination rcl … rcn …"
  exit 1
fi
if ! grep -q "rdif" "${LINGO}"; then
  echo "FAIL: ${LINGO} missing rdif reaction coordinate"
  exit 1
fi
# Stale ratio / distance-only lingo after switching to difference.
if grep -Eq 'name[[:space:]]+r_nc|name[[:space:]]+rrat|define[[:space:]]+rrat[[:space:]]+ratio|scombination' "${LINGO}"; then
  echo "FAIL: ${LINGO} still has old r_nc/rrat/scombination RC — wipe output_dir and re-run"
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
