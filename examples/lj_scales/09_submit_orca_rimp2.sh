#!/usr/bin/env bash
# Step 09 — submit / collect RI-MP2 def2-TZVP EnGrad labels for the joint bank.
#
# Cluster-side only. Two modes:
#   LJ_ORCA_MODE=submit   (default)  make_orca_array.py → print sbatch line
#   LJ_ORCA_MODE=collect             collect_orca_array.py → labeled NPZ
#
# Keywords match the production ORCA template:
#   RI-MP2 def2-TZVP def2-TZVP/C def2/J RIJCOSX TightSCF EnGrad
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
lj_scales_banner

echo "=== 09: ORCA RI-MP2/def2-TZVP (${LJ_ORCA_MODE}) ==="

GEOMS="${LJ_JOINT_GEOMS}"
RUN_DIR="${LJ_ORCA_RUN_DIR}"
LABELED="${LJ_JOINT_LABELED}"
KEYWORDS="${LJ_ORCA_KEYWORDS}"

if [[ ! -f "${GEOMS}" ]]; then
  echo "ERROR: geometry bank missing: ${GEOMS}" >&2
  echo "       Run 08_build_joint_geoms.sh first." >&2
  exit 2
fi

case "${LJ_ORCA_MODE}" in
  submit)
    mkdir -p "${RUN_DIR}"
    uv run python scripts/make_orca_array.py \
      --data "${GEOMS}" \
      --out "${RUN_DIR}" \
      --keywords "${KEYWORDS}" \
      --nprocs "${LJ_ORCA_NPROCS}" \
      --maxcore "${LJ_ORCA_MAXCORE}" \
      --chunk "${LJ_ORCA_CHUNK}" \
      --throttle "${LJ_ORCA_THROTTLE}" \
      --walltime "${LJ_ORCA_WALLTIME}" \
      --partition "${LJ_ORCA_PARTITION}" \
      --module "${LJ_ORCA_MODULE}"
    cat <<EOF

09: array prepared under ${RUN_DIR}
    sbatch ${RUN_DIR}/run_array.sh

When the array finishes:
    LJ_ORCA_MODE=collect bash examples/lj_scales/09_submit_orca_rimp2.sh
EOF
    ;;
  collect)
    if [[ ! -d "${RUN_DIR}/dat" ]]; then
      echo "ERROR: no ${RUN_DIR}/dat — submit the array first." >&2
      exit 2
    fi
    mkdir -p "$(dirname "${LJ_JOINT_LABELED_BASE}")"
    # collect_orca_array writes ${base}_{train,valid,test}.npz from --out stem.
    uv run python scripts/collect_orca_array.py \
      --run-dir "${RUN_DIR}" \
      --source "${GEOMS}" \
      --out "${LJ_JOINT_LABELED_BASE}.npz" \
      --method "RI-MP2/def2-TZVP"
    if [[ ! -f "${LABELED}" ]]; then
      echo "ERROR: expected train split at ${LABELED}" >&2
      exit 3
    fi
    echo "09: OK  ${LABELED} (+ valid/test siblings)"
    ;;
  *)
    echo "ERROR: LJ_ORCA_MODE must be 'submit' or 'collect' (got '${LJ_ORCA_MODE}')" >&2
    exit 2
    ;;
esac
