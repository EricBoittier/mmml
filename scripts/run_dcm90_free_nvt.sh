#!/usr/bin/env bash
# Free-space NVT: MLpot SD → heating → Hoover CPT equil + MMFP sphere.
#
# Sphere radii (initial guess before minimization):
#   packmol sphere R ≈ 18 * (90/60)^(1/3) ≈ 21 Å for DCM:90.
#   MMFP droff (flat-bottom) after COM centering:
#     R_fb = 1 + sqrt(x_max^2 + y_max^2 + z_max^2)
#   Recompute from mini CRD (see estimate_droff_from_crd.py below) and rerun with --fb-rad.
#
# Prerequisites:
#   - Rebuilt libcharmm.so (source has max_Nml=50000; stale lib segfaults on DCM:90):
#       ./scripts/rebuild_charmm_mlpot.sh
#   - Launch under OpenMPI mpirun (libcharmm.so is MPI-linked; plain python segfaults
#     after JAX GPU warmup on gpu nodes):
#       ./scripts/mmml-charmm-mpirun.sh md-system ...
#   - MMML_CKPT or examples/ckpts_json/DESdimers_params.json
#   - packmol on PATH (mmml/generate/packmol or module)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Initial sphere guess; tighten droff after mini using the formula above (~30–33 Å).
PACKMOL_R="${PACKMOL_R:-21.0}"
FB_RAD="${FB_RAD:-32.0}"
COMPOSITION="${1:-DCM:90}"
if [[ $# -gt 0 ]]; then
  shift
fi

COMPOSITION_TAG="$(
  printf '%s' "$COMPOSITION" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+//g'
)"
RUN_NAME="${COMPOSITION_TAG}_nvt"

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
#exec "$MPIRUN" 

uv run mmml md-system \
  --setup free_nvt \
  --backend pycharmm \
  --composition "$COMPOSITION" \
  --output-dir "artifacts/pycharmm_mlpot/$RUN_NAME" \
  --job-name "$RUN_NAME" \
  --md-stages mini,heat,equi \
  --free-space \
  --packmol-sphere \
  --packmol-radius "$PACKMOL_R" \
  --packmol-tolerance 1.0 \
  --flat-bottom-radius "$FB_RAD" --dynamics-overlap-min-distance 0.4 \
  --flat-bottom-k 0.01 \
  --temperature 50.0 --bonded-mm-mini --bonded-mm-mini-after mini,heat,equi --bonded-mm-mini-steps 500 \
  --dt-fs 0.5 \
  --ps-heat 1 \
  --ps-equi 1 \
  --dcd-nsavc 40 \
  --dyn-nprint 40 --ml-switch-width 0.1 --mm-switch-width 3.0 --charmm-sd-steps 2000 --charmm-abnr-steps 200  \
  --ml-batch-size 2256 \
  --skip-energy-show \
  --seed 123 \
  "$@"
