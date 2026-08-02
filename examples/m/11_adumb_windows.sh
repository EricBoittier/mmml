#!/usr/bin/env bash
# Independent full-range ADUMB replicas, each seeded from a different NPZ frame
# across the reaction coordinate xi = r(Cl-C)/r(C-N).
#
# ADUMB is adaptive: every run flattens the FULL window xi in [0.125, 5.0] on its
# own, so these are independent replicas (not sub-range windows) — robust (no
# 'reaction coordinate out of range' aborts) and each yields a full PMF. Seeding
# from frames spread across xi de-correlates the replicas and helps fill the
# xi in [0.9, 1.1] TS gap that the dataset is missing. Combine by averaging the
# per-window PMFs (ADUMB-WUNI.DAT) or pooling histograms via RUNI.
#
#   source examples/m/_env.sh
#   bash examples/m/11_adumb_windows.sh              # run all 6 sequentially
#   DRY_RUN=1 bash examples/m/11_adumb_windows.sh    # print per-window commands
#   XIS="1.0 1.5" bash examples/m/11_adumb_windows.sh # subset
#
# Cluster: run DRY_RUN=1 to get one self-contained `mmml md-system` command per
# window, then submit each as its own sbatch job to sample the replicas in
# parallel (they share nothing but the read-only NPZ/checkpoint).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/m/_env.sh"
cd "${ROOT}"

CFG="${CFG:-${ROOT}/examples/m/yaml/adumb_nc_distance.yaml}"
XIS="${XIS:-0.5 0.8 1.0 1.5 2.0 3.0}"
BASE_OUT="${BASE_OUT:-${ARTIFACTS_DIR}/adumb_windows}"
DRY_RUN="${DRY_RUN:-0}"

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

echo "=== ADUMB full-range replicas seeded across xi: ${XIS} ==="
echo "    config=${CFG}"
echo "    base output=${BASE_OUT}"

rc_all=0
declare -a DONE=()
for xi in ${XIS}; do
  tag="xi_${xi}"
  OUT="${BASE_OUT}/${tag}"
  SOLUTE="${OUT}/solute_${tag}.pdb"
  mkdir -p "${OUT}"
  # Drop stale per-window state so a re-run starts clean (matches 09 script).
  rm -rf "${OUT}/.packmol_cache" "${OUT}/packmol_cluster" "${OUT}/pretreat" "${OUT}/cleanup"
  rm -f "${OUT}/stage_summary.json" \
    "${OUT}/next_run.yaml" "${OUT}/next_run.sh" "${OUT}/next_run.command" \
    "${OUT}/pycharmm_pre_dynamics_lingo.inp"

  # Centered (default) TS-relative seed geometry for this window.
  uv run python examples/m/07_export_solute_pdb.py --xi "${xi}" -o "${SOLUTE}"

  # Full-range ADUMB from that geometry; --no-packmol keeps the exact NPZ frame.
  # Seed preservation (SEED_PRESERVE=1, default): skip the full-CGenFF MM pre-min
  # and isolated-monomer PhysNet mini so a broken/dissociated seed is not relaxed
  # back to the reactant geometry before dynamics (the hybrid ML BFGS is kept).
  CMD=(uv run mmml md-system
    --config "${CFG}"
    --output-dir "${OUT}"
    --composition "${SOLUTE}"
    --from-pdb "${SOLUTE}"
    --no-packmol)
  if [[ "${SEED_PRESERVE:-1}" == "1" ]]; then
    CMD+=(--charmm-sd-steps 0 --charmm-abnr-steps 0 --no-monomer-physnet-mini)
  fi

  echo "--- window ${tag} -> ${OUT}"
  printf '    '; printf '%q ' "${CMD[@]}"; echo
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  set +e
  "${CMD[@]}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL: window ${tag} exited ${rc}"
    rc_all=1
    continue
  fi
  if [[ -f "${OUT}/ADUMB-WUNI.DAT" || -f "${OUT}/adumb-wuni.dat" ]]; then
    DONE+=("${tag}")
  else
    echo "WARN: window ${tag} produced no ADUMB-WUNI.DAT"
    rc_all=1
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN: printed ${XIS} window commands (nothing executed)"
  exit 0
fi

echo "=== windows with ADUMB output: ${DONE[*]:-none} ==="
if [[ "${rc_all}" -ne 0 ]]; then
  echo "FAIL: one or more windows did not complete"
  exit 1
fi
echo "PASS: all windows -> ${BASE_OUT}/xi_*/ADUMB-WUNI.DAT"
