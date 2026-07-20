#!/usr/bin/env bash
# TIP3 liquid-density Ewald smoke: Packmol liquid + CHARMM MM pretreat, then
# short hybrid heat + NVE with Ewald (self term omitted).
#
# Default geometry is TIP3:90 in 30 Å (~1 g/cm³). A cubic grid + random
# rotations at this density leaves hybrid FIRE stuck at fmax≈5–6 eV/Å and
# MLpot SD can spike to 1e5+ GRMS — Packmol + MM pretreat avoid that.
#
# Important: do NOT resume a failed tip3_*_smoke next_run / baseline.res.
# Wipe the output dir and restart from this script.
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

echo "== TIP3:${N_MOL} / ${BOX_SIZE} Å Packmol → MM pretreat → heat ${PS_HEAT} ps → NVE ${PS_NVE} ps =="
echo "  checkpoint: $CKPT"
echo "  output:     $OUT_DIR"
echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
echo "  mm-charge:  $MM_CHARGE_MODE"
echo "  density:    prep ladder OFF (ewald wiring smoke; not density campaign)"
echo "  builder:    Packmol cube (not lattice grid)"
echo "  repair:     --no-monomer-physnet-mini (keep liquid packing)"

# Packmol liquid in --box-size, then CHARMM MM mini/heat before MLpot.
# Grid builders at ~1 g/cm³ leave every water "stressed" (fmax~5 eV/Å); hybrid
# FIRE cannot reach the 2 eV/Å pre-heat gate and MLpot SD spikes catastrophically.
# --no-monomer-physnet-mini: vacuum PhysNet on waters wrecks H-bond packing.
# --density-prep-mode off: avoid FIRE/BFGS thrash on this smoke path.
set +e
mmml md-system \
  --backend pycharmm \
  --setup pycharmm_full \
  --md-stages mini,heat,nve \
  --composition "TIP3:${N_MOL}" \
  --packmol \
  --packmol-placement cube \
  --box-size "$BOX_SIZE" \
  --rebuild-packmol \
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
  --charmm-mm-pretreat-heat-nstep 4000 \
  --charmm-mm-pretreat-ps-equi 0.5 \
  --charmm-mm-pretreat-mini-sd 200 \
  --charmm-mm-pretreat-mini-abnr 500 \
  --charmm-sd-steps 100 \
  --charmm-abnr-steps 200 \
  --fire-min-steps 400 \
  --fire-min-maxstep 0.05 \
  "$@"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). See $OUT_DIR/next_run.command if present." >&2
  echo "Do not resume next_run/baseline from a failed gate — wipe and restart:" >&2
  echo "  rm -rf $OUT_DIR && re-run this script" >&2
  exit "$rc"
fi
echo "Done. Artifacts under $OUT_DIR"
