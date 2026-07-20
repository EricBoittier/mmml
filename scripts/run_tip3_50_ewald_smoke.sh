#!/usr/bin/env bash
# TIP3:50 on a cubic grid in a 30 Å PBC box (random monomer rotations),
# then short heat + NVE with hybrid Ewald (self term omitted).
#
# Usage (gpu09 / CHARMM+JAX env):
#   export CKPT=/path/to/physnet_or_spooky.json
#   ./scripts/run_tip3_50_ewald_smoke.sh
#
# Optional env:
#   OUT_DIR, SEED, PS_HEAT, PS_NVE, TEMP_K, DT_FS, MM_CHARGE_MODE

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:?set CKPT to a portable PhysNet/Spooky JSON (or Orbax dir)}"
OUT_DIR="${OUT_DIR:-./scratch/tip3_50_ewald_smoke}"
SEED="${SEED:-42}"
PS_HEAT="${PS_HEAT:-2.0}"
PS_NVE="${PS_NVE:-2.0}"
TEMP_K="${TEMP_K:-300}"
DT_FS="${DT_FS:-0.5}"
MM_CHARGE_MODE="${MM_CHARGE_MODE:-fixed}"

mkdir -p "$OUT_DIR"

echo "== TIP3:50 / 30 Å grid + random rotations → heat ${PS_HEAT} ps → NVE ${PS_NVE} ps =="
echo "  checkpoint: $CKPT"
echo "  output:     $OUT_DIR"
echo "  lr-solver:  ewald --ewald-omit-self"
echo "  mm-charge:  $MM_CHARGE_MODE"

# Grid builder (skips Packmol): even COM lattice in --box-size, SO(3) from --seed.
# Stages: mini → heat → nve. Ewald self omitted for MIC/non-Ewald-trained models.
set +e
mmml md-system \
  --backend pycharmm \
  --setup pycharmm_full \
  --md-stages mini,heat,nve \
  --composition TIP3:50 \
  --builder liquid \
  --no-packmol \
  --box-size 30 \
  --mlpot-pbc \
  --seed "$SEED" \
  --checkpoint "$CKPT" \
  --output-dir "$OUT_DIR" \
  --temperature "$TEMP_K" \
  --dt-fs "$DT_FS" \
  --ps-heat "$PS_HEAT" \
  --ps-nve "$PS_NVE" \
  --include-mm \
  --mm-charge-mode "$MM_CHARGE_MODE" \
  --lr-solver ewald \
  --ewald-omit-self \
  --mm-nonbond-mode jax_mic \
  --ml-switch-width 1.5 \
  --mm-switch-on 6.0 \
  --mm-switch-width 5.0 \
  "$@"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). See $OUT_DIR/next_run.command if present." >&2
  exit "$rc"
fi
echo "Done. Artifacts under $OUT_DIR"
