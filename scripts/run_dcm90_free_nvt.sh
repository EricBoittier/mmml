#!/usr/bin/env bash
# DCM:90 vacuum (free-space) NVT: MLpot SD → heating → Hoover CPT equil + MMFP sphere.
#
# Sphere radii (initial guess before minimization):
#   packmol sphere R ≈ 18 * (90/60)^(1/3) ≈ 21 Å  (see packmol/packmol_sphere_dcm90.inp)
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

MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"
exec "$MPIRUN" md-system \
  --setup free_nvt \
  --backend pycharmm \
  --composition DCM:90 \
  --output-dir artifacts/pycharmm_mlpot/dcm90_nvt \
  --job-name dcm90_nvt \
  --md-stages mini,heat,equi \
  --free-space \
  --packmol-sphere \
  --packmol-radius "$PACKMOL_R" \
  --packmol-tolerance 2.0 \
  --flat-bottom-radius "$FB_RAD" \
  --flat-bottom-k 1.0 \
  --temperature 300 \
  --dt-fs 0.25 \
  --ps-heat 20 \
  --ps-equi 50 \
  --dcd-nsavc 400 \
  --dyn-nprint 500 \
  --ml-batch-size 256 \
  --skip-energy-show \
  --seed 123 \
  "$@"
