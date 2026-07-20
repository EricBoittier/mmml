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
# Default stages: mini,heat,equi
#   - heat: Hoover NVT (pmass=0), gentle FIRSTT→FINALT ramp (default 40→200 K)
#   - equi: CPT NpT (pmass>0) at bath temperature (default 200 K)
# Skip heat with MD_STAGES=mini,equi (not recommended after cold CPT ECHECK).
#
# DYNA list cadence: DYN_INBFRQ / DYN_IMGFRQ default 25 (was CHARMM 50). More
# frequent IMAGE/NB/MLpot updates during early barostat motion; still cheap vs
# every step. Override with DYN_INBFRQ=10 DYN_IMGFRQ=10 for tighter lists.
#
# Heat resilience: --no-echeck-heat on the heat leg only; EQUI keeps ECHECK.
# Do not resume next_run / --no-echeck-heat after a CPT abort.
#
# Pass: exit 0 with equi restart present; L still ~30 Å.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:?set CKPT to PhysNet portable JSON}"
BOX_OPT_OUT="${BOX_OPT_OUT:-./scratch/tip3_physnet_ewald_ir/tip3_30A_box_opt}"
LIQUID_DIR="${LIQUID_DIR:-$BOX_OPT_OUT/liquid_box}"
OPT_DIR="${OPT_DIR:-$BOX_OPT_OUT/box_pressure_opt}"
OUT_DIR="${OUT_DIR:-$BOX_OPT_OUT/npt_charmm}"
MM_CHARGE_MODE="${MM_CHARGE_MODE:-fixed}"
# Staging bath (K). Keep ≤200 until CPT is stable; raise TEMP_K later for 300 K.
TEMP_K="${TEMP_K:-200}"
# Gentle heat ramp start (K). Default ~0.2×TEMP_K; override with HEAT_FIRSTT.
HEAT_FIRSTT="${HEAT_FIRSTT:-}"
TARGET_P_ATM="${TARGET_P_ATM:-1.0}"
SEED="${SEED:-42}"
DT_FS="${DT_FS:-0.25}"
PS_HEAT="${PS_HEAT:-2.0}"
PS_EQUI="${PS_EQUI:-2.0}"
WIPE="${WIPE:-1}"
# NVT ramp then CPT (avoid cold start straight into 300 K barostat).
MD_STAGES="${MD_STAGES:-mini,heat,equi}"
N_HEAT_SEGMENTS="${N_HEAT_SEGMENTS:-8}"
# Tighter than CHARMM default 50 during early NpT / heat.
DYN_INBFRQ="${DYN_INBFRQ:-25}"
DYN_IMGFRQ="${DYN_IMGFRQ:-25}"
# Tier-1 local multi-GPU PhysNet chunks (not spatial MPI). Default 1.
ML_GPU_COUNT="${ML_GPU_COUNT:-1}"
ML_BATCH_SIZE="${ML_BATCH_SIZE:-}"

if [[ -z "$HEAT_FIRSTT" ]]; then
  HEAT_FIRSTT="$(
    uv run python -c "print(f'{float(\"$TEMP_K\") * 0.2:.1f}')"
  )"
fi

ML_ARGS=(--ml-gpu-count "$ML_GPU_COUNT")
if [[ -n "$ML_BATCH_SIZE" ]]; then
  ML_ARGS+=(--ml-batch-size "$ML_BATCH_SIZE")
fi
HEAT_ARGS=()
if [[ ",$MD_STAGES," == *",heat,"* ]]; then
  HEAT_ARGS=(
    --ps-heat "$PS_HEAT"
    --heat-thermostat hoover
    --heat-firstt "$HEAT_FIRSTT"
    --heat-finalt "$TEMP_K"
    --no-echeck-heat
    --n-heat-segments "$N_HEAT_SEGMENTS"
    # One DYNA per heat segment (no mid-segment velocity redraw).
    --heat-overlap-segment-boundary-only
  )
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
echo "  stages:     $MD_STAGES  (CPT equi @ ${TARGET_P_ATM} atm; pmass>0)"
echo "  equi:       ${PS_EQUI} ps   dt=${DT_FS} fs"
echo "  dyn lists:  inbfrq=${DYN_INBFRQ}  imgfrq=${DYN_IMGFRQ}"
if [[ ",$MD_STAGES," == *",heat,"* ]]; then
  echo "  heat:       ${PS_HEAT} ps Hoover NVT (pmass=0) ramp ${HEAT_FIRSTT}→${TEMP_K} K (${N_HEAT_SEGMENTS} segments)"
fi
echo "  lr-solver:  ewald --ewald-omit-self --mlpot-pbc"
echo "  ml-gpus:    $ML_GPU_COUNT  batch=${ML_BATCH_SIZE:-auto}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +e
mmml md-system \
  --backend pycharmm \
  --setup pbc_npt \
  --md-stages "$MD_STAGES" \
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
  --ps-equi "$PS_EQUI" \
  --npt-thermostat hoover \
  --dyn-inbfrq "$DYN_INBFRQ" \
  --dyn-imgfrq "$DYN_IMGFRQ" \
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
  --dynamics-intra-rescue-sd-steps "${INTRA_RESCUE_SD_STEPS:-400}" \
  "${HEAT_ARGS[@]}" \
  "${ML_ARGS[@]}" \
  "$@"
rc=$?
set -e

# Partial equi*.res after ECHECK/ABNORMAL is not success (unlike box_opt PRRTE).
if [[ "$rc" -ne 0 ]]; then
  echo "FAILED (exit $rc). Do not resume next_run / --no-echeck-heat from a CPT abort." >&2
  echo "  Wipe and re-run after git pull:" >&2
  echo "  rm -rf $OUT_DIR && re-run this script" >&2
  exit "$rc"
fi

echo "Pass: CHARMM CPT NpT smoke under $OUT_DIR"
echo "  Check: equi restart, pressure_tensor log if written, final L near ${BOX_SIDE} Å"
