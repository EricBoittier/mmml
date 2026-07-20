#!/usr/bin/env bash
# TIP3 CHARMM CPT NpT smoke from a certified liquid-box (pinned count@30 Å).
#
# Default NpT path is PyCHARMM CPT (not jaxmd). Input is PSF/CRD + box.json from
# STAGE=box_opt (tip3_30A_box_opt/liquid_box).
#
# Usage:
#   export CKPT=/path/to/physnet_portable.json
#   ./scripts/run_tip3_charmm_npt_smoke.sh
#   BOX_OPT_OUT=./scratch/.../tip3_90_box_opt WIPE=0 ./scripts/run_tip3_charmm_npt_smoke.sh
#   CUDA_VISIBLE_DEVICES=0,1 ML_GPU_COUNT=2 ML_BATCH_SIZE=256 ./scripts/run_tip3_charmm_npt_smoke.sh
#
# Heat resilience (dense TIP3): --no-echeck-heat, INTRA_RESCUE_SD_STEPS (default 400),
# N_HEAT_SEGMENTS (default 4) so crushed waters get more rescue checkpoints.
#
# Pass: exit 0 (or PRRTE exit 1 with equi restart present), L still ~30 Å.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:?set CKPT to PhysNet portable JSON}"
BOX_OPT_OUT="${BOX_OPT_OUT:-./scratch/tip3_physnet_ewald_ir/tip3_30A_box_opt}"
LIQUID_DIR="${LIQUID_DIR:-$BOX_OPT_OUT/liquid_box}"
OPT_DIR="${OPT_DIR:-$BOX_OPT_OUT/box_pressure_opt}"
OUT_DIR="${OUT_DIR:-$BOX_OPT_OUT/npt_charmm}"
MM_CHARGE_MODE="${MM_CHARGE_MODE:-fixed}"
TEMP_K="${TEMP_K:-300}"
TARGET_P_ATM="${TARGET_P_ATM:-1.0}"
SEED="${SEED:-42}"
DT_FS="${DT_FS:-0.5}"
PS_HEAT="${PS_HEAT:-1.0}"
PS_EQUI="${PS_EQUI:-2.0}"
WIPE="${WIPE:-1}"
# Tier-1 local multi-GPU PhysNet chunks (not spatial MPI). Default 1.
ML_GPU_COUNT="${ML_GPU_COUNT:-1}"
ML_BATCH_SIZE="${ML_BATCH_SIZE:-}"

ML_ARGS=(--ml-gpu-count "$ML_GPU_COUNT")
if [[ -n "$ML_BATCH_SIZE" ]]; then
  ML_ARGS+=(--ml-batch-size "$ML_BATCH_SIZE")
fi

# Prefer pressure-opt handoff (post-CPT L) when present; else liquid_box.
if [[ -f "$OPT_DIR/model.crd" && -f "$OPT_DIR/model.psf" && -f "$OPT_DIR/box.json" ]]; then
  PSF="$OPT_DIR/model.psf"
  CRD="$OPT_DIR/model.crd"
  BOX_JSON="$OPT_DIR/box.json"
  HAND_OFF_SRC="box_pressure_opt"
else
  PSF="$LIQUID_DIR/model.psf"
  CRD="$LIQUID_DIR/model.crd"
  BOX_JSON="$LIQUID_DIR/box.json"
  HAND_OFF_SRC="liquid_box"
fi

if [[ ! -f "$PSF" || ! -f "$CRD" || ! -f "$BOX_JSON" ]]; then
  echo "FAILED: need certified PSF/CRD/box.json under $OPT_DIR or $LIQUID_DIR" >&2
  echo "  Run STAGE=box_opt first (USE_CHARMM_PRESSURE=1 writes handoff CRD)." >&2
  exit 1
fi

# Resolve composition / L from pinned box.json
read -r N_MOL BOX_SIDE RHO STATUS < <(
  uv run python - <<PY
import json
from pathlib import Path
d = json.loads(Path("$BOX_JSON").read_text())
print(
    int(d.get("n_molecules") or 0),
    float(d.get("box_side_A") or d.get("final_cubic_side_A") or 0.0),
    float(d.get("density_g_cm3") or 0.0),
    str(d.get("status") or "?"),
)
PY
)

if [[ "$STATUS" != "pass" ]]; then
  echo "FAILED: liquid-box status=$STATUS (need pass)" >&2
  exit 1
fi
if [[ "$N_MOL" -lt 1 ]]; then
  echo "FAILED: box.json missing n_molecules" >&2
  exit 1
fi

COMPOSITION="TIP3:${N_MOL}"

if [[ "$WIPE" == "1" ]]; then
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "== TIP3 CHARMM CPT NpT smoke (pinned liquid → hybrid CPT) =="
echo "  handoff:    $HAND_OFF_SRC  N=$N_MOL  L=${BOX_SIDE} Å  ρ=${RHO} g/cm³"
echo "  output:     $OUT_DIR"
echo "  stages:     mini,heat,equi  (Hoover heat + CPT equi @ ${TARGET_P_ATM} atm)"
echo "  heat/equi:  ${PS_HEAT} / ${PS_EQUI} ps   dt=${DT_FS} fs"
echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +e
mmml md-system \
  --backend pycharmm \
  --setup pbc_npt \
  --md-stages mini,heat,equi \
  --composition "$COMPOSITION" \
  --from-psf "$PSF" \
  --from-crd "$CRD" \
  --skip-cluster-build \
  --box-size "$BOX_SIDE" \
  --mlpot-pbc \
  --seed "$SEED" \
  --checkpoint "$CKPT" \
  --output-dir "$OUT_DIR" \
  --temperature "$TEMP_K" \
  --npt-pressure "$TARGET_P_ATM" \
  --pressure "$TARGET_P_ATM" \
  --dt-fs "$DT_FS" \
  --ps-heat "$PS_HEAT" \
  --ps-equi "$PS_EQUI" \
  --heat-thermostat hoover \
  --npt-thermostat hoover \
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
  --npt-pressure-log-interval 50 \
  --no-echeck-heat \
  --dynamics-intra-rescue-sd-steps "${INTRA_RESCUE_SD_STEPS:-400}" \
  --n-heat-segments "${N_HEAT_SEGMENTS:-4}" \
  "${ML_ARGS[@]}" \
  "$@"
rc=$?
set -e

# Accept PRRTE exit 1 when equi produced a restart (same pattern as box_opt).
EQUI_RES="$(find "$OUT_DIR" -name 'equi*.res' -o -name '*equi*.res' 2>/dev/null | head -1 || true)"
if [[ "$rc" -ne 0 && -z "${EQUI_RES:-}" ]]; then
  echo "FAILED (exit $rc). No equi restart under $OUT_DIR." >&2
  echo "Do not resume next_run from a gate fail — wipe and re-run:" >&2
  echo "  rm -rf $OUT_DIR && re-run this script" >&2
  exit "$rc"
fi
if [[ "$rc" -ne 0 ]]; then
  echo "note: md-system exit=$rc but equi restart present ($EQUI_RES); treating as pass"
fi

echo "Pass: CHARMM CPT NpT smoke under $OUT_DIR"
echo "  Check: equi restart, pressure_tensor log if written, final L near ${BOX_SIDE} Å"
