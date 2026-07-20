#!/usr/bin/env bash
# Staged TIP3 PhysNet campaign: PBC FD → dimer Ewald scan → tip3_50 smoke →
# production jaxmd NVE → IR analysis (analyze_water_nve_h5.py).
#
# Prerequisites (gpu09): synced tree, OpenMPI/CHARMM, JAX GPU, CKPT set.
#
#   export CKPT=/path/to/test-f41c04c0-..._epoch-251_portable.json
#   ./scripts/run_tip3_physnet_ewald_ir_campaign.sh              # all stages
#   STAGE=fd ./scripts/run_tip3_physnet_ewald_ir_campaign.sh     # one stage
#   STAGE=scan,smoke ./scripts/run_tip3_physnet_ewald_ir_campaign.sh
#
# Stages: fd | scan | smoke | prod | analyze | all
#
# Pass/fail (quick):
#   fd      — fd_force_max_abs_diff_eVA < 0.05
#   scan    — scan_1d.npz written under SCAN_OUT
#   smoke   — tip3_50 heat+NVE exit 0
#   prod    — jaxmd H5 exists; NVE finishes
#   analyze — ir_spectrum.png + OH power with peak in 2800–3600 cm^-1 (PhysNet)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${CKPT:?set CKPT to PhysNet portable JSON (charges=False → use fixed MM)}"
OUT_ROOT="${OUT_ROOT:-./scratch/tip3_physnet_ewald_ir}"
MM_CHARGE_MODE="${MM_CHARGE_MODE:-fixed}"
STAGE="${STAGE:-all}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$ROOT/scripts/mmml-charmm-mpirun.sh}"

# Geometry / timing knobs
BOX_SMOKE="${BOX_SMOKE:-30}"
N_SMOKE="${N_SMOKE:-50}"
PS_HEAT_SMOKE="${PS_HEAT_SMOKE:-2.0}"
PS_NVE_SMOKE="${PS_NVE_SMOKE:-2.0}"

# Production IR (jaxmd H5 → analyze_water_nve_h5). Defaults match spooky-scale
# sampling (dt 0.25 fs, record every 10 → frame_dt 2.5 fs; Nyquist ~6670 cm^-1).
# TIP3:50 @ 30 Å is dilute (~0.055 g/cm³); for denser liquid set N_PROD/BOX_PROD.
BOX_PROD="${BOX_PROD:-30}"
N_PROD="${N_PROD:-50}"
PS_PROD="${PS_PROD:-50}"
DT_FS_PROD="${DT_FS_PROD:-0.25}"
STEPS_PER_REC="${STEPS_PER_REC:-10}"
TEMP_K="${TEMP_K:-300}"
SEED="${SEED:-42}"

SCAN_OUT="${SCAN_OUT:-$OUT_ROOT/dimer_scan}"
SMOKE_OUT="${SMOKE_OUT:-$OUT_ROOT/tip3_50_smoke}"
PROD_OUT="${PROD_OUT:-$OUT_ROOT/tip3_${N_PROD}_nve}"
FD_OUT="${FD_OUT:-$OUT_ROOT/pbc_fd_tip3.json}"
ANALYZE_OUT="${ANALYZE_OUT:-$OUT_ROOT/analysis}"

mkdir -p "$OUT_ROOT"

_want() {
  local name="$1"
  [[ "$STAGE" == "all" ]] && return 0
  [[ ",$STAGE," == *",$name,"* ]] && return 0
  return 1
}

echo "== TIP3 PhysNet Ewald IR campaign =="
echo "  CKPT=$CKPT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  STAGE=$STAGE  mm-charge=$MM_CHARGE_MODE  lr=ewald --ewald-omit-self"

# --- 1) TIP3 PBC FD ---------------------------------------------------------
if _want fd; then
  echo ""
  echo "=== [fd] mode-check --pbc-fd TIP3 + ewald omit-self ==="
  if ! mmml mode-check --pbc-fd \
    --residue TIP3 \
    --n-molecules 10 \
    --checkpoint "$CKPT" \
    --lr-solver ewald \
    --ewald-omit-self \
    --mm-charge-mode "$MM_CHARGE_MODE" \
    --output "$FD_OUT"
  then
    echo "FAILED: mode-check --pbc-fd (TIP3)." >&2
    exit 1
  fi
  if [[ ! -f "$FD_OUT" ]]; then
    echo "FAILED: missing $FD_OUT" >&2
    exit 1
  fi
  uv run python - <<PY
import json
from pathlib import Path
p = Path("$FD_OUT")
d = json.loads(p.read_text())
mx = float(d["fd_force_max_abs_diff_eVA"])
print(f"FD max |ΔF| = {mx:.6g} eV/Å  (pass < 0.05)")
raise SystemExit(0 if mx < 0.05 else 1)
PY
fi

# --- 2) TIP3:2 hybrid-native Ewald dimer scan ------------------------------
if _want scan; then
  echo ""
  echo "=== [scan] TIP3:2 pbc_hybrid_ewald_omit_self ==="
  "$MPIRUN" python scripts/scan_mlpot_dimer_2d_pycharmm.py \
    --composition TIP3:2 \
    --scan-1d \
    --scan-tag pbc_hybrid_ewald_omit_self \
    --box-size 30 \
    --mlpot-pbc \
    --lr-solver ewald \
    --ewald-omit-self \
    --mm-charge-mode "$MM_CHARGE_MODE" \
    --checkpoint "$CKPT" \
    --scan-2d-min 3.5 \
    --scan-2d-max 14.0 \
    --scan-2d-steps 12 \
    --mm-switch-on 6.0 \
    --mm-switch-width 5.0 \
    --ml-switch-width 1.5 \
    --output-dir "$SCAN_OUT" \
    --skip-energy-show \
    --seed "$SEED"
  test -n "$(find "$SCAN_OUT" -name 'scan_1d.npz' 2>/dev/null | head -1)"
  echo "scan NPZ under $SCAN_OUT"
fi

# --- 3) tip3_50 PyCHARMM smoke ---------------------------------------------
if _want smoke; then
  echo ""
  echo "=== [smoke] TIP3:${N_SMOKE} PyCHARMM heat+NVE ==="
  OUT_DIR="$SMOKE_OUT" \
  PS_HEAT="$PS_HEAT_SMOKE" \
  PS_NVE="$PS_NVE_SMOKE" \
  MM_CHARGE_MODE="$MM_CHARGE_MODE" \
  TEMP_K="$TEMP_K" \
  SEED="$SEED" \
  ./scripts/run_tip3_50_ewald_smoke.sh
fi

# --- 4) Production jaxmd NVE (H5 for IR) ------------------------------------
if _want prod; then
  echo ""
  echo "=== [prod] TIP3:${N_PROD} jaxmd NVE ${PS_PROD} ps (IR trajectory) ==="
  echo "  box=${BOX_PROD} Å  dt=${DT_FS_PROD} fs  record/${STEPS_PER_REC} → frame_dt=$(
    awk -v d="$DT_FS_PROD" -v s="$STEPS_PER_REC" 'BEGIN{printf "%.3f", d*s}'
  ) fs"
  mmml md-system \
    --backend jaxmd \
    --setup pbc_nve \
    --composition "TIP3:${N_PROD}" \
    --builder liquid \
    --no-packmol \
    --box-size "$BOX_PROD" \
    --seed "$SEED" \
    --checkpoint "$CKPT" \
    --output-dir "$PROD_OUT" \
    --temperature "$TEMP_K" \
    --dt-fs "$DT_FS_PROD" \
    --ps "$PS_PROD" \
    --steps-per-recording "$STEPS_PER_REC" \
    --include-mm \
    --mm-charge-mode "$MM_CHARGE_MODE" \
    --lr-solver ewald \
    --ewald-omit-self \
    --ml-switch-width 1.5 \
    --mm-switch-on 6.0 \
    --mm-switch-width 5.0
  H5="$(find "$PROD_OUT" -name '*.h5' | head -1 || true)"
  if [[ -z "$H5" ]]; then
    echo "FAILED: no HDF5 under $PROD_OUT" >&2
    exit 1
  fi
  echo "H5=$H5" | tee "$PROD_OUT/h5_path.txt"
fi

# --- 5) IR + OH bond analysis ----------------------------------------------
if _want analyze; then
  echo ""
  echo "=== [analyze] IR / OH power from jaxmd H5 ==="
  if [[ -f "$PROD_OUT/h5_path.txt" ]]; then
    H5="$(sed -n 's/^H5=//p' "$PROD_OUT/h5_path.txt" | head -1)"
  else
    H5="$(find "$PROD_OUT" -name '*.h5' | head -1 || true)"
  fi
  if [[ -z "${H5:-}" || ! -f "$H5" ]]; then
    echo "FAILED: set STAGE=prod first or point PROD_OUT at a finished NVE." >&2
    exit 1
  fi
  uv run python scripts/analyze_water_nve_h5.py \
    --h5 "$H5" \
    --box-A "$BOX_PROD" \
    --output-dir "$ANALYZE_OUT" \
    --ir-temperature-K "$TEMP_K"
  echo "Artifacts under $ANALYZE_OUT"
  echo "  Check: ir_spectrum.png, oh_bond_power_spectra.png, summary.json"
  echo "  Pass (PhysNet): OH power peak ~2800–3600 cm^-1 (not ~40 cm^-1)."
fi

echo ""
echo "Campaign stage(s) done. OUT_ROOT=$OUT_ROOT"
