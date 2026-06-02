#!/usr/bin/env bash
# DCM:90 free-space heating from an existing MLpot mini (CGENFF + hybrid MMML MLpot).
#
# Hybrid force field (both on by default in DecomposedMlpotModel):
#   doML=True   — PhysNet monomer/dimer ML (USER term via MLpot)
#   doMM=True   — CHARMM MM inter-monomer handoff (complementary taper)
#
# Uses mini PSF/CRD under OUT_DIR; does not rebuild Packmol cluster.
#
# Example (on gpu08):
#   ./scripts/run_dcm90_heat_from_mini.sh
#   FB_RAD=33 ./scripts/run_dcm90_heat_from_mini.sh --ps-heat 20
#
# Prerequisites: same as scripts/run_dcm90_free_nvt.sh (mpirun wrapper, libcharmm, checkpoint).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-artifacts/pycharmm_mlpot/dcm90_nvt}"
TAG="${TAG:-dcm_90}"
FB_RAD="${FB_RAD:-32.0}"
PS_HEAT="${PS_HEAT:-20}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$REPO_ROOT/scripts/mmml-charmm-mpirun.sh}"

MINI_PSF="$OUT_DIR/mini_full_mlpot_${TAG}.psf"
MINI_CRD="$OUT_DIR/mini_full_mlpot_${TAG}.crd"
if [[ ! -f "$MINI_PSF" || ! -f "$MINI_CRD" ]]; then
  echo "Missing mini artifacts: $MINI_PSF and/or $MINI_CRD" >&2
  echo "Run ./scripts/run_dcm90_free_nvt.sh --md-stages mini first." >&2
  exit 1
fi

exec "$MPIRUN" md-system \
  --setup free_nvt \
  --backend pycharmm \
  --composition DCM:90 \
  --output-dir "$OUT_DIR" \
  --job-name dcm90_heat \
  --md-stage heat \
  --skip-cluster-build \
  --from-psf "$MINI_PSF" \
  --from-crd "$MINI_CRD" \
  --no-save-vmd-topology \
  --free-space \
  --flat-bottom-radius "$FB_RAD" \
  --flat-bottom-k 1.0 \
  --temperature 300 \
  --dt-fs 0.25 \
  --ps-heat "$PS_HEAT" \
  --dcd-nsavc 400 \
  --dyn-nprint 500 \
  --ml-batch-size 256 \
  --skip-energy-show \
  --mm-switch-on 7.0 \
  --mm-switch-width 5.0 \
  --ml-switch-width 0.1 \
  --seed 123 \
  "$@"
