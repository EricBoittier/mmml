#!/usr/bin/env bash
# Run the trainable-LJ-scales ladder.
#
# Steps 00, 03, 04 are self-contained and run in seconds on CPU — no dataset, no
# CHARMM, no GPU. They are the teaching core and always run.
#
# Steps 01, 02, 05, 07 need real inputs and are skipped with a message when those
# are absent, so `bash run_all.sh` is always safe to type.
#
#   LJ_FULL=1                run the expensive steps (01, 05, 07) too
#   LJ_DEVICE=gpu            use the GPU (recommended for 05)
#   LJ_EPOCHS=50             cheap first training pass
#   LJ_DATASET=/path.npz     input QM data (PSF-ordered)
#   LJ_JOINT=1               ACO+DCM joint path (steps 08–11; cluster ORCA) —
#                            not driven by this script; see README
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/examples/lj_scales/_env.sh"
cd "${ROOT}"
DIR="examples/lj_scales"
lj_scales_banner

echo "=== trainable LJ scales ladder ==="
if [[ "${LJ_JOINT}" == "1" ]]; then
  cat <<EOF
LJ_JOINT=1: run the joint path manually (ORCA is cluster-side):
  bash ${DIR}/08_build_joint_geoms.sh
  bash ${DIR}/09_submit_orca_rimp2.sh && LJ_ORCA_MODE=collect bash ${DIR}/09_submit_orca_rimp2.sh
  bash ${DIR}/10_merge_prepare_joint.sh
  bash ${DIR}/05_train.sh && uv run python ${DIR}/06_inspect_scales.py
  bash ${DIR}/11_liquid_boxes.sh
  bash ${DIR}/07_deploy_md.sh
EOF
fi

uv run python "${DIR}/00_check_env.py"

if [[ -f "${LJ_DATASET}" ]]; then
  uv run python "${DIR}/02_inspect_dataset.py"
else
  echo "SKIP 02: no dataset at ${LJ_DATASET}"
fi

# Always run: these are the conceptual core and need nothing external.
uv run python "${DIR}/03_gradient_demo.py"
uv run python "${DIR}/04_miniature_fit.py"

if [[ "${LJ_FULL:-0}" != "1" ]]; then
  cat <<EOF

=== fast ladder complete (00, 02, 03, 04) ===
The expensive steps were skipped. To run them:

  LJ_FULL=1 LJ_DEVICE=gpu bash ${DIR}/run_all.sh

or individually:

  bash ${DIR}/01_prepare_dataset.sh    # minutes  (CGenFF assignment)
  bash ${DIR}/05_train.sh              # hours    (wants a GPU)
  uv run python ${DIR}/06_inspect_scales.py
  bash ${DIR}/07_deploy_md.sh          # needs PyCHARMM
EOF
  exit 0
fi

bash "${DIR}/01_prepare_dataset.sh"
uv run python "${DIR}/02_inspect_dataset.py"
bash "${DIR}/05_train.sh"
uv run python "${DIR}/06_inspect_scales.py"
bash "${DIR}/07_deploy_md.sh"

echo "=== ALL LJ SCALE STEPS PASSED ==="
