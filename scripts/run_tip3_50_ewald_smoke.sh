#!/usr/bin/env bash
# TIP3 liquid-density Ewald smoke: cubic grid + random rotations, then short
# heat + NVE with hybrid Ewald (self term omitted).
#
# Default geometry is TIP3:90 in 30 Å (~1 g/cm³). The old TIP3:50/30 Å grid
# is dilute (~0.055 g/cm³) and thrash the density-prep / FIRE ladder.
#
# Important: do NOT resume a failed tip3_*_smoke next_run that already ran
# isolated vacuum PhysNet on every water — wipe the output dir and restart.
#
# Usage (gpu09 / CHARMM+JAX env):
#   export CKPT=/path/to/physnet_or_spooky.json
#   ./scripts/run_tip3_50_ewald_smoke.sh
#
# Optional env:
#   OUT_DIR, SEED, PS_HEAT, PS_NVE, TEMP_K, DT_FS, MM_CHARGE_MODE
#   N_MOL (default 90), BOX_SIZE (default 30)

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
N_MOL="${N_MOL:-90}"
BOX_SIZE="${BOX_SIZE:-30}"

mkdir -p "$OUT_DIR"

echo "== TIP3:${N_MOL} / ${BOX_SIZE} Å grid + random rotations → heat ${PS_HEAT} ps → NVE ${PS_NVE} ps =="
echo "  checkpoint: $CKPT"
echo "  output:     $OUT_DIR"
echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
echo "  mm-charge:  $MM_CHARGE_MODE"
echo "  density:    prep ladder OFF (ewald wiring smoke; not density campaign)"
echo "  repair:     --no-monomer-physnet-mini (keep liquid packing)"

# Grid builder (skips Packmol): even COM lattice in --box-size, SO(3) from --seed.
# Stages: mini → heat → nve. Ewald self omitted for MIC/non-Ewald-trained models.
# --mlpot-pbc: hybrid calculator gets the cell (required for lr_solver=ewald).
# --density-prep-mode off: avoid FIRE/BFGS thrash on this smoke path.
# --no-monomer-physnet-mini: vacuum PhysNet on all waters wrecks H-bond packing
#   and trips the pre-heat max-|F| gate (~7 eV/Å) after "successful" monomer FIRE.
# --charmm-mm-pretreat: short MM heat to settle contacts before hybrid dynamics.
set +e
mmml md-system \
  --backend pycharmm \
  --setup pycharmm_full \
  --md-stages mini,heat,nve \
  --composition "TIP3:${N_MOL}" \
  --builder liquid \
  --no-packmol \
  --box-size "$BOX_SIZE" \
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
  --density-prep-mode off \
  --no-density-prep-ladder \
  --no-mc-density-equalize \
  --no-monomer-physnet-mini \
  --charmm-mm-pretreat \
  --charmm-mm-pretreat-heat-nstep 2000 \
  --charmm-sd-steps 200 \
  --charmm-abnr-steps 500 \
  --fire-min-steps 400 \
  --fire-min-maxstep 0.05 \
  "$@"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). See $OUT_DIR/next_run.command if present." >&2
  echo "If this dir already has a broken baseline from isolated PhysNet repair, wipe it:" >&2
  echo "  rm -rf $OUT_DIR && re-run this script" >&2
  exit "$rc"
fi
echo "Done. Artifacts under $OUT_DIR"
