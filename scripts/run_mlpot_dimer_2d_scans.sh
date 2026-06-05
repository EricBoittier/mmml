#!/usr/bin/env bash
# Run PyCHARMM MLpot 2D trimer scans (d01 × d02) for checkpoint(s) and composition(s).
#
# Each run writes:
#   <OUT_ROOT>/<checkpoint_basename>/<composition_tag>/scan_2d.npz
#
# Prerequisites: same as mmml md-system --backend pycharmm (OpenMPI, libcharmm, MMML_CKPT).
#
# Examples:
#   export MMML_CKPT=/path/to/dcm1-.../ckpts/dcm1-...
#   ./scripts/run_mlpot_dimer_2d_scans.sh
#
#   COMPOSITIONS="DCM:3 ACO:3" CHECKPOINTS="/ckpt/a /ckpt/b" ./scripts/run_mlpot_dimer_2d_scans.sh
#
#   PACKMOL_R=8 ./scripts/run_mlpot_dimer_2d_scans.sh DCM:3

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${MMML_CKPT:-}" && -z "${CHECKPOINTS:-}" ]]; then
  echo "Set MMML_CKPT or CHECKPOINTS (space-separated checkpoint paths)." >&2
  exit 1
fi

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
OUT_ROOT="${OUT_ROOT:-artifacts/pycharmm_mlpot/dimer_2d_scan}"
SCAN_2D_MIN="${SCAN_2D_MIN:-3.0}"
SCAN_2D_MAX="${SCAN_2D_MAX:-10.0}"
SCAN_2D_STEPS="${SCAN_2D_STEPS:-15}"
ANGLE_02_DEG="${ANGLE_02_DEG:-60.0}"
PACKMOL_R="${PACKMOL_R:-8.0}"

if [[ -n "${CHECKPOINTS:-}" ]]; then
  # shellcheck disable=SC2206
  CKPT_LIST=($CHECKPOINTS)
else
  CKPT_LIST=("$MMML_CKPT")
fi

if [[ $# -ge 1 ]]; then
  # shellcheck disable=SC2206
  COMPOSITIONS=("$@")
elif [[ -n "${COMPOSITIONS:-}" ]]; then
  # shellcheck disable=SC2206
  COMPOSITIONS=($COMPOSITIONS)
else
  COMPOSITIONS=("DCM:3")
fi

for ckpt in "${CKPT_LIST[@]}"; do
  for comp in "${COMPOSITIONS[@]}"; do
    echo "=== 2D scan: checkpoint=${ckpt} composition=${comp} ==="
    "$MPIRUN" python scripts/scan_mlpot_dimer_2d_pycharmm.py \
      --composition "$comp" \
      --checkpoint "$ckpt" \
      --output-dir "$OUT_ROOT" \
      --packmol-sphere \
      --packmol-radius "$PACKMOL_R" \
      --packmol-tolerance 1.0 \
      --scan-2d-min "$SCAN_2D_MIN" \
      --scan-2d-max "$SCAN_2D_MAX" \
      --scan-2d-steps "$SCAN_2D_STEPS" \
      --angle-02-deg "$ANGLE_02_DEG" \
      --skip-energy-show \
      --seed 123
  done
done

echo "Done. NPZ files under ${OUT_ROOT}/"
