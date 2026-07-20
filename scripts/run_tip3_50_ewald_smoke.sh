#!/usr/bin/env bash
# TIP3 liquid-density Ewald smoke: short hybrid heat (+ optional NVE) with Ewald
# (self term omitted).
#
# Two input modes:
#   1) CERTIFIED_BOX_DIR — handoff from box_pressure_opt (or liquid_box):
#        model.psf + model.crd + box.json → fixed-L hybrid smoke (no Packmol).
#   2) Default Packmol TIP3:N_MOL @ BOX_SIZE + CHARMM MM pretreat.
#
# Density goal after a successful hybrid heat is CHARMM CPT NpT (not hybrid NVE).
# With CONTINUE_TO_NPT=1 (default for certified handoff), a heat*.res under OUT_DIR
# triggers scripts/run_tip3_charmm_npt_smoke.sh even if NVE failed later.
#
# Important: do NOT resume a failed tip3_*_smoke next_run / baseline.res.
# Wipe the output dir and restart from this script (or run STAGE=npt directly).
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
#   MD_STAGES (default mini,heat,nve; use mini,heat to stop before NVE)
#   CONTINUE_TO_NPT (default 1 for certified, 0 for Packmol) — run CHARMM CPT
#   BOX_OPT_OUT — parent of liquid_box/box_pressure_opt for the NPT stage
#   NPT_OUT, PS_HEAT_NPT, PS_EQUI_NPT, TARGET_P_ATM, WIPE_NPT

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
MD_STAGES="${MD_STAGES:-mini,heat,nve}"
# Set after USE_CERTIFIED is known (below); allow explicit override.
CONTINUE_TO_NPT="${CONTINUE_TO_NPT:-}"
TARGET_P_ATM="${TARGET_P_ATM:-1.0}"
PS_HEAT_NPT="${PS_HEAT_NPT:-1.0}"
PS_EQUI_NPT="${PS_EQUI_NPT:-2.0}"
WIPE_NPT="${WIPE_NPT:-1}"

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

# Density goal: after hybrid heat, CHARMM CPT NpT (default on for certified).
if [[ -z "$CONTINUE_TO_NPT" ]]; then
  if [[ "$USE_CERTIFIED" == "1" ]]; then
    CONTINUE_TO_NPT=1
  else
    CONTINUE_TO_NPT=0
  fi
fi

_find_heat_restart() {
  local root="$1"
  # Prefer explicit heat*.res; fall back to any *heat*.res under OUT_DIR.
  find "$root" \( -name 'heat.res' -o -name 'heat*.res' -o -name '*heat*.res' \) \
    -type f 2>/dev/null | head -1 || true
}

_run_charmm_npt_from_handoff() {
  local box_opt_out npt_out
  if [[ -n "${BOX_OPT_OUT:-}" ]]; then
    box_opt_out="$BOX_OPT_OUT"
  elif [[ -n "${CERTIFIED_BOX_DIR:-}" ]]; then
    # CERTIFIED is .../box_pressure_opt or .../liquid_box → parent is BOX_OPT_OUT
    box_opt_out="$(cd "$(dirname "$CERTIFIED_BOX_DIR")" && pwd)"
  else
    echo "CONTINUE_TO_NPT=1 but neither BOX_OPT_OUT nor CERTIFIED_BOX_DIR set" >&2
    return 1
  fi
  npt_out="${NPT_OUT:-$box_opt_out/npt_charmm}"
  echo ""
  echo "== CONTINUE_TO_NPT: CHARMM CPT from $box_opt_out → $npt_out =="
  BOX_OPT_OUT="$box_opt_out" \
  OUT_DIR="$npt_out" \
  TARGET_P_ATM="$TARGET_P_ATM" \
  TEMP_K="$TEMP_K" \
  SEED="$SEED" \
  PS_HEAT="$PS_HEAT_NPT" \
  PS_EQUI="$PS_EQUI_NPT" \
  MM_CHARGE_MODE="$MM_CHARGE_MODE" \
  WIPE="$WIPE_NPT" \
  ./scripts/run_tip3_charmm_npt_smoke.sh
}

if [[ "$USE_CERTIFIED" == "1" ]]; then
  echo "== TIP3:${N_MOL} certified handoff @ L=${BOX_SIZE} Å → stages ${MD_STAGES} (heat ${PS_HEAT} ps / NVE ${PS_NVE} ps) =="
  echo "  from-psf:   $FROM_PSF"
  echo "  from-crd:   $FROM_CRD"
  echo "  box.json:   ${BOX_JSON:-"(none; using BOX_SIZE=$BOX_SIZE)"}"
  echo "  checkpoint: $CKPT"
  echo "  output:     $OUT_DIR"
  echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
  echo "  mm-charge:  $MM_CHARGE_MODE"
  echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "  mode:       fixed-L hybrid smoke (no Packmol rebuild)"
  echo "  after-heat: CONTINUE_TO_NPT=$CONTINUE_TO_NPT (CHARMM CPT when heat*.res exists)"

  set +e
  mmml md-system \
    --backend pycharmm \
    --setup pycharmm_full \
    --md-stages "$MD_STAGES" \
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
  echo "== TIP3:${N_MOL} / ${BOX_SIZE} Å Packmol → MM pretreat → stages ${MD_STAGES} (heat ${PS_HEAT} ps / NVE ${PS_NVE} ps) =="
  echo "  checkpoint: $CKPT"
  echo "  output:     $OUT_DIR"
  echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
  echo "  mm-charge:  $MM_CHARGE_MODE"
  echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "  density:    prep ladder OFF (ewald wiring smoke; not density campaign)"
  echo "  builder:    Packmol cube (not lattice grid)"
  echo "  repair:     shared PhysNet water template (preserve liquid COM/orientation)"
  echo "  after-heat: CONTINUE_TO_NPT=$CONTINUE_TO_NPT"

  # Packmol liquid in --box-size, then CHARMM MM mini/heat before MLpot.
  set +e
  mmml md-system \
    --backend pycharmm \
    --setup pycharmm_full \
    --md-stages "$MD_STAGES" \
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

HEAT_RES="$(_find_heat_restart "$OUT_DIR")"

if [[ "$CONTINUE_TO_NPT" == "1" && -n "${HEAT_RES:-}" ]]; then
  echo "Hybrid heat restart present: $HEAT_RES"
  if [[ "$rc" -ne 0 ]]; then
    echo "note: md-system exit=$rc but heat left a restart — moving on to CHARMM CPT NpT (density goal)"
  fi
  _run_charmm_npt_from_handoff
  echo "Done. Hybrid smoke under $OUT_DIR; CHARMM NpT under ${NPT_OUT:-$(dirname "${CERTIFIED_BOX_DIR:-$OUT_DIR}")/npt_charmm}"
  exit 0
fi

if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). See $OUT_DIR/next_run.command if present." >&2
  if [[ -n "${HEAT_RES:-}" ]]; then
    echo "Heat restart exists ($HEAT_RES). Density goal is CHARMM CPT — prefer:" >&2
    echo "  BOX_OPT_OUT=<parent of box_pressure_opt> STAGE=npt ./scripts/run_tip3_physnet_ewald_ir_campaign.sh" >&2
  else
    echo "Do not resume next_run/baseline from a failed gate — wipe and restart:" >&2
    echo "  rm -rf $OUT_DIR && re-run this script" >&2
    echo "Or skip hybrid NVE and run NpT from certified handoff:" >&2
    echo "  STAGE=npt BOX_OPT_OUT=./scratch/tip3_physnet_ewald_ir/tip3_90_box_opt \\" >&2
    echo "    ./scripts/run_tip3_physnet_ewald_ir_campaign.sh" >&2
  fi
  exit "$rc"
fi
echo "Done. Artifacts under $OUT_DIR"
if [[ "$CONTINUE_TO_NPT" == "1" && -z "${HEAT_RES:-}" ]]; then
  echo "note: CONTINUE_TO_NPT=1 but no heat*.res under $OUT_DIR — skipped CHARMM NpT" >&2
fi
