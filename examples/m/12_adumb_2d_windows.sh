#!/usr/bin/env bash
# Independent 2D ADUMB replicas seeded across the (r_ClC, r_CN) plane.
#
# Companion to 11_adumb_windows.sh (1D ξ) for the 2D map (10_adumb_clc_cn_2d.yaml).
# ADUMB is adaptive and starts with zero bias, so it does NOT hold a window — it
# relaxes into whichever basin the seed sits in and flattens outward from there.
# So each grid point seeds a *basin* and each run maps that basin; the union
# tiles the plane. Seeds come from real NPZ frames (07_export --rcl/--rcn), and
# SEED_PRESERVE (default on) skips the pre-min steps that would otherwise reform
# the C-Cl bond and collapse a broken/dissociated seed before dynamics.
#
# Grid points are "rcl,rcn" pairs in Å. Default spans the reaction:
#   reactant (1.8,3.5)  near-TS (2.2,2.0)  product (3.8,1.57)  dissociated (6.0,1.5)
#
#   source examples/m/_env.sh
#   bash examples/m/12_adumb_2d_windows.sh                 # run all points sequentially
#   DRY_RUN=1 bash examples/m/12_adumb_2d_windows.sh       # print one md-system cmd/point
#   GRID="3.8,1.57 6.0,1.5" bash examples/m/12_adumb_2d_windows.sh   # subset
#
# Cluster: DRY_RUN=1 gives one self-contained `mmml md-system` command per point;
# submit each as its own sbatch job (they share only the read-only NPZ/checkpoint).
# Combine the per-point ADUMB-WUNI.DAT 2D histograms via WHAM.
#
# Caveat: the bundled dataset has no frames at ξ∈[0.9,1.1] and only sparse,
# high-energy data at r_ClC≈2-3 Å, so the ML PES at the barrier is extrapolation.
# This maps the basins; a trustworthy ΔG‡ needs training data at the TS.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/adumb_clc_cn_2d.yaml}"
GRID="${GRID:-1.8,3.5 2.2,2.0 3.8,1.57 6.0,1.5}"
BASE_OUT="${BASE_OUT:-${ARTIFACTS_DIR}/adumb_2d_windows}"
DRY_RUN="${DRY_RUN:-0}"
SEED_PRESERVE="${SEED_PRESERVE:-1}"

if ! uv run python -c "import pycharmm" >/dev/null 2>&1; then
  echo "SKIP: PyCHARMM not importable"
  exit 0
fi

if [[ -z "${MMML_CGENFF_EXTRA_RTF:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_RTF unset — CH3CL will not be in CGenFF"
fi
if [[ -z "${MMML_CGENFF_EXTRA_PRM:-}" ]]; then
  echo "WARN: MMML_CGENFF_EXTRA_PRM unset — CG331–CLGA1 bond/angle may be missing"
fi

echo "=== 2D ADUMB replicas seeded across (r_ClC, r_CN): ${GRID} ==="
echo "    config=${CFG}"
echo "    base output=${BASE_OUT}  seed_preserve=${SEED_PRESERVE}"

rc_all=0
declare -a DONE=()
for pt in ${GRID}; do
  rcl="${pt%,*}"
  rcn="${pt#*,}"
  if [[ "${rcl}" == "${pt}" || -z "${rcl}" || -z "${rcn}" ]]; then
    echo "FAIL: bad grid point '${pt}' — want 'rcl,rcn' (Å), e.g. 3.8,1.57"
    rc_all=1
    continue
  fi
  tag="rcl${rcl}_rcn${rcn}"
  OUT="${BASE_OUT}/${tag}"
  SOLUTE="${OUT}/solute_${tag}.pdb"
  mkdir -p "${OUT}"
  # Drop stale per-window state so a re-run starts clean (matches 09/11).
  rm -rf "${OUT}/.packmol_cache" "${OUT}/packmol_cluster" "${OUT}/pretreat" "${OUT}/cleanup"
  rm -f "${OUT}/stage_summary.json" \
    "${OUT}/next_run.yaml" "${OUT}/next_run.sh" "${OUT}/next_run.command" \
    "${OUT}/pycharmm_pre_dynamics_lingo.inp"

  # Centered seed nearest this (rcl, rcn) point.
  uv run python examples/m/07_export_solute_pdb.py --rcl "${rcl}" --rcn "${rcn}" -o "${SOLUTE}"

  CMD=(uv run mmml md-system
    --config "${CFG}"
    --output-dir "${OUT}"
    --composition "${SOLUTE}"
    --from-pdb "${SOLUTE}"
    --no-packmol)
  # Preserve the seeded reaction coordinate: skip the full-CGenFF MM pre-min and
  # isolated-monomer PhysNet mini (both reform C-Cl); keep the hybrid ML BFGS.
  if [[ "${SEED_PRESERVE}" == "1" ]]; then
    CMD+=(--charmm-sd-steps 0 --charmm-abnr-steps 0 --no-monomer-physnet-mini)
  fi

  echo "--- point ${tag} -> ${OUT}"
  printf '    '; printf '%q ' "${CMD[@]}"; echo
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  set +e
  "${CMD[@]}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL: point ${tag} exited ${rc}"
    rc_all=1
    continue
  fi
  if [[ -f "${OUT}/ADUMB-WUNI.DAT" || -f "${OUT}/adumb-wuni.dat" ]]; then
    DONE+=("${tag}")
  else
    echo "WARN: point ${tag} produced no ADUMB-WUNI.DAT"
    rc_all=1
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: printed grid commands (nothing executed)"
  exit 0
fi

echo "=== points with ADUMB output: ${DONE[*]:-none} ==="
if [[ "${rc_all}" -ne 0 ]]; then
  echo "FAIL: one or more grid points did not complete"
  exit 1
fi
echo "PASS: all points -> ${BASE_OUT}/rcl*_rcn*/ADUMB-WUNI.DAT"
