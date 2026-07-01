#!/usr/bin/env bash
# DCM and ACO dimer COM scans across MM nonbond modes and long-range Coulomb solvers.
#
# Writes one NPZ per configuration under:
#   ${OUT_ROOT}/<checkpoint>/<composition_tag>/<scan_tag>/scan_1d.npz
#
# Prerequisites: MMML_CKPT, OpenMPI, libcharmm (scripts/mmml-charmm-mpirun.sh).
#
# Examples:
#   export MMML_CKPT=/path/to/dcm_ckpt
#   ./scripts/run_dcm_aco_dimer_lr_scans.sh
#
#   COMPOSITIONS="DCM:2" SKIP_PERIODIC=1 ./scripts/run_dcm_aco_dimer_lr_scans.sh
#
#   OUT_ROOT=artifacts/dimer_lr_scans BOX_SIZE=40 ./scripts/run_dcm_aco_dimer_lr_scans.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${MMML_CKPT:-}" ]]; then
  echo "Set MMML_CKPT to a DCM/ACO PhysNet checkpoint." >&2
  exit 1
fi

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/dimer_lr_scans}"
BOX_SIZE="${BOX_SIZE:-36.0}"
SCAN_MIN="${SCAN_MIN:-3.5}"
SCAN_MAX="${SCAN_MAX:-14.0}"
SCAN_STEPS="${SCAN_STEPS:-12}"
PACKMOL_R="${PACKMOL_R:-10.0}"

if [[ $# -ge 1 ]]; then
  # shellcheck disable=SC2206
  COMPOSITIONS=("$@")
elif [[ -n "${COMPOSITIONS:-}" ]]; then
  # shellcheck disable=SC2206
  COMPOSITIONS=($COMPOSITIONS)
else
  COMPOSITIONS=("DCM:2" "ACO:2")
fi

SKIP_PERIODIC="${SKIP_PERIODIC:-0}"

_common_scan_args=(
  --checkpoint "$MMML_CKPT"
  --output-dir "$OUT_ROOT"
  --scan-1d
  --scan-2d-min "$SCAN_MIN"
  --scan-2d-max "$SCAN_MAX"
  --scan-2d-steps "$SCAN_STEPS"
  --mm-switch-on 8.0
  --mm-switch-width 5.0
  --ml-switch-width 1.5
  --packmol-sphere
  --packmol-radius "$PACKMOL_R"
  --packmol-tolerance 1.0
  --skip-energy-show
  --seed 42
)

_run_scan() {
  local comp="$1"
  local tag="$2"
  shift 2
  echo ""
  echo "=== ${comp}  scan_tag=${tag} ==="
  "$MPIRUN" python scripts/scan_mlpot_dimer_2d_pycharmm.py \
    --composition "$comp" \
    --scan-tag "$tag" \
    "${_common_scan_args[@]}" \
    "$@"
}

_have_nvalchemiops() {
  uv run python -c "from mmml.interfaces.pycharmmInterface.long_range_backend import have_nvalchemiops_pme; raise SystemExit(0 if have_nvalchemiops_pme() else 1)" 2>/dev/null
}

_have_scafacos() {
  uv run python -c "from mmml.interfaces.pycharmmInterface.long_range_backend import have_scafacos; raise SystemExit(0 if have_scafacos() else 1)" 2>/dev/null
}

for comp in "${COMPOSITIONS[@]}"; do
  _run_scan "$comp" "vacuum_mic" --free-space --lr-solver mic

  _run_scan "$comp" "pbc_mic" \
    --box-size "$BOX_SIZE" --mlpot-pbc --lr-solver mic

  for method in ewald pme p3m; do
    _run_scan "$comp" "pbc_jax_pme_${method}" \
      --box-size "$BOX_SIZE" --mlpot-pbc \
      --lr-solver jax_pme --jax-pme-method "$method"
  done

  _run_scan "$comp" "pbc_jax_pme_ewald_no_disp" \
    --box-size "$BOX_SIZE" --mlpot-pbc \
    --lr-solver jax_pme --jax-pme-method ewald --no-jax-pme-dispersion

  if [[ "$SKIP_PERIODIC" == "1" ]]; then
    continue
  fi

  _run_scan "$comp" "pbc_periodic_external_jax_pme_pme" \
    --box-size "$BOX_SIZE" --mlpot-pbc \
    --mm-nonbond-mode periodic_external \
    --lr-solver jax_pme --jax-pme-method pme

  if _have_nvalchemiops; then
    _run_scan "$comp" "pbc_periodic_external_nvalchemiops" \
      --box-size "$BOX_SIZE" --mlpot-pbc \
      --mm-nonbond-mode periodic_external \
      --lr-solver nvalchemiops_pme
  else
    echo "Skipping nvalchemiops_pme (not installed) for ${comp}"
  fi

  if _have_scafacos; then
    _run_scan "$comp" "pbc_periodic_external_scafacos_ewald" \
      --box-size "$BOX_SIZE" --mlpot-pbc \
      --mm-nonbond-mode periodic_external \
      --lr-solver scafacos --scafacos-method ewald
  else
    echo "Skipping scafacos (libfcs not found) for ${comp}"
  fi
done

echo ""
echo "Done. NPZ files under ${OUT_ROOT}/"
echo "Plot: uv run python scripts/plot_dimer_lr_scan_compare.py --root ${OUT_ROOT}"
