#!/usr/bin/env bash
# TIP3 liquid-density Ewald smoke: short hybrid heat + NVE with Ewald
# (self term omitted).
#
# Two input modes:
#   1) CERTIFIED_BOX_DIR — handoff from box_pressure_opt (or liquid_box):
#        model.psf + model.crd + box.json → fixed-L hybrid smoke (no Packmol).
#   2) Default Packmol TIP3:N_MOL @ BOX_SIZE + CHARMM MM pretreat.
#
# Important: do NOT resume a failed tip3_*_smoke next_run / baseline.res.
# Wipe the output dir and restart from this script.
#
# Usage (gpu09 / CHARMM+JAX env):
#   export CKPT=/path/to/physnet_or_spooky.json
#   ./scripts/run_tip3_50_ewald_smoke.sh
#   CERTIFIED_BOX_DIR=./scratch/.../tip3_30A_box_opt/box_pressure_opt \
#     ./scripts/run_tip3_50_ewald_smoke.sh
#
# Optional env:
#   OUT_DIR, SEED, PS_HEAT, PS_NVE, TEMP_K, DT_FS, MM_CHARGE_MODE
#   N_MOL (default 90), BOX_SIZE (default 30)
#   CERTIFIED_BOX_DIR, FROM_PSF, FROM_CRD, BOX_JSON
#   ML_GPU_COUNT (default 1), ML_BATCH_SIZE (default auto: 256 on GPU for n>=40)
#   CUDA_VISIBLE_DEVICES — set to the GPU ids (e.g. 0,1 with ML_GPU_COUNT=2)

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
CERTIFIED_BOX_DIR="${CERTIFIED_BOX_DIR:-}"
FROM_PSF="${FROM_PSF:-}"
FROM_CRD="${FROM_CRD:-}"
BOX_JSON="${BOX_JSON:-}"
# Multi-GPU PhysNet chunks (Tier-1 local pmap). Default 1; use 2 with CUDA_VISIBLE_DEVICES=0,1.
ML_GPU_COUNT="${ML_GPU_COUNT:-1}"
# Chunk size for PhysNet; empty → auto (256 GPU / 64 CPU for n>=40).
ML_BATCH_SIZE="${ML_BATCH_SIZE:-}"

mkdir -p "$OUT_DIR"

ML_ARGS=(--ml-gpu-count "$ML_GPU_COUNT")
if [[ -n "$ML_BATCH_SIZE" ]]; then
  ML_ARGS+=(--ml-batch-size "$ML_BATCH_SIZE")
fi

# Resolve certified handoff (box_pressure_opt preferred over liquid_box).
if [[ -n "$CERTIFIED_BOX_DIR" ]]; then
  FROM_PSF="${FROM_PSF:-$CERTIFIED_BOX_DIR/model.psf}"
  FROM_CRD="${FROM_CRD:-$CERTIFIED_BOX_DIR/model.crd}"
  BOX_JSON="${BOX_JSON:-$CERTIFIED_BOX_DIR/box.json}"
fi

USE_CERTIFIED=0
if [[ -n "$FROM_PSF" && -n "$FROM_CRD" ]]; then
  if [[ ! -f "$FROM_PSF" || ! -f "$FROM_CRD" ]]; then
    echo "FAILED: certified handoff missing PSF/CRD:" >&2
    echo "  FROM_PSF=$FROM_PSF" >&2
    echo "  FROM_CRD=$FROM_CRD" >&2
    exit 1
  fi
  USE_CERTIFIED=1
  if [[ -n "$BOX_JSON" && -f "$BOX_JSON" ]]; then
    read -r N_MOL BOX_SIZE STATUS < <(
      uv run python - <<PY
import json
from pathlib import Path
d = json.loads(Path("$BOX_JSON").read_text())
print(
    int(d.get("n_molecules") or 0),
    float(d.get("box_side_A") or d.get("final_cubic_side_A") or 0.0),
    str(d.get("status") or "?"),
)
PY
    )
    if [[ "$STATUS" != "pass" ]]; then
      echo "FAILED: certified box.json status=$STATUS (need pass)" >&2
      exit 1
    fi
    if [[ "$N_MOL" -lt 1 ]]; then
      echo "FAILED: box.json missing n_molecules" >&2
      exit 1
    fi
  fi
fi

if [[ "$USE_CERTIFIED" == "1" ]]; then
  echo "== TIP3:${N_MOL} certified handoff @ L=${BOX_SIZE} Å → heat ${PS_HEAT} ps → NVE ${PS_NVE} ps =="
  echo "  from-psf:   $FROM_PSF"
  echo "  from-crd:   $FROM_CRD"
  echo "  box.json:   ${BOX_JSON:-"(none; using BOX_SIZE=$BOX_SIZE)"}"
  echo "  checkpoint: $CKPT"
  echo "  output:     $OUT_DIR"
  echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
  echo "  mm-charge:  $MM_CHARGE_MODE"
  echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "  mode:       fixed-L hybrid smoke (no Packmol rebuild)"

  set +e
  mmml md-system \
    --backend pycharmm \
    --setup pycharmm_full \
    --md-stages mini,heat,nve \
    --composition "TIP3:${N_MOL}" \
    --from-psf "$FROM_PSF" \
    --from-crd "$FROM_CRD" \
    --skip-cluster-build \
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
    --no-charmm-pre-minimize \
    --charmm-sd-steps 50 \
    --charmm-abnr-steps 100 \
    --fire-min-steps 200 \
    --fire-min-maxstep 0.05 \
    "${ML_ARGS[@]}" \
    "$@"
  rc=$?
  set -e
else
  echo "== TIP3:${N_MOL} / ${BOX_SIZE} Å Packmol → MM pretreat → heat ${PS_HEAT} ps → NVE ${PS_NVE} ps =="
  echo "  checkpoint: $CKPT"
  echo "  output:     $OUT_DIR"
  echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
  echo "  mm-charge:  $MM_CHARGE_MODE"
  echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "  density:    prep ladder OFF (ewald wiring smoke; not density campaign)"
  echo "  builder:    Packmol cube (not lattice grid)"
  echo "  repair:     shared PhysNet water template (preserve liquid COM/orientation)"

  # Packmol liquid in --box-size, then CHARMM MM mini/heat before MLpot.
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
    --monomer-physnet-mini \
    --charmm-mm-pretreat \
    --charmm-mm-pretreat-heat-nstep 4000 \
    --charmm-mm-pretreat-ps-equi 0.5 \
    --charmm-mm-pretreat-mini-sd 200 \
    --charmm-mm-pretreat-mini-abnr 500 \
    --charmm-sd-steps 100 \
    --charmm-abnr-steps 200 \
    --fire-min-steps 400 \
    --fire-min-maxstep 0.05 \
    "${ML_ARGS[@]}" \
    "$@"
  rc=$?
  set -e
fi

if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). See $OUT_DIR/next_run.command if present." >&2
  echo "Do not resume next_run/baseline from a failed gate — wipe and restart:" >&2
  echo "  rm -rf $OUT_DIR && re-run this script" >&2
  exit "$rc"
fi
echo "Done. Artifacts under $OUT_DIR"
