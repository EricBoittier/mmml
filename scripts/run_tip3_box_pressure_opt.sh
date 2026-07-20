#!/usr/bin/env bash
# TIP3 CHARMM-default box pressure opt prep:
#   liquid-box (MM certify at target density) → pressure MC + 1D refine → box.json
#
# Offline default uses a synthetic P∝1/L³ model calibrated to the certified side
# (pipeline smoke without a live CHARMM virial loop). On gpu09, set
# USE_CHARMM_PRESSURE=1 once the CHARMM pressure adapter session is wired for
# this entry point.
#
# Density note: TIP3:90 @ 30 Å is ~0.1 g/cm³, not 1 g/cm³ (~900 waters @ 30 Å,
# or L≈14 Å for 90 waters). This script sizes the cell from N + target density
# (--box-auto density). To fill a fixed cube instead:
#   BOX_MODE=count BOX_SIZE=30 ./scripts/run_tip3_box_pressure_opt.sh
#
# Usage:
#   ./scripts/run_tip3_box_pressure_opt.sh
#   N_MOL=90 TARGET_DENSITY=1.0 OUT_DIR=./scratch/tip3_box_opt ./scripts/run_tip3_box_pressure_opt.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-./scratch/tip3_physnet_ewald_ir/tip3_90_box_opt}"
N_MOL="${N_MOL:-90}"
BOX_SIZE="${BOX_SIZE:-30}"
BOX_MODE="${BOX_MODE:-density}"   # density | count
TARGET_P_ATM="${TARGET_P_ATM:-1.0}"
TEMP_K="${TEMP_K:-300}"
SEED="${SEED:-42}"
TARGET_DENSITY="${TARGET_DENSITY:-1.0}"

mkdir -p "$OUT_DIR"
LIQUID_DIR="$OUT_DIR/liquid_box"
OPT_DIR="$OUT_DIR/box_pressure_opt"

echo "== TIP3 box pressure opt (CHARMM-default NpT prep) =="
echo "  liquid-box → $LIQUID_DIR"
echo "  pressure opt → $OPT_DIR"
echo "  target P = ${TARGET_P_ATM} atm   T = ${TEMP_K} K   ρ_target = ${TARGET_DENSITY} g/cm³"
echo "  BOX_MODE=$BOX_MODE"

echo ""
echo "=== [1/2] mmml liquid-box (MM certify) ==="
# Do not pass --box-size together with --target-density-g-cm3 unless N matches ρ:
# liquid-box holds L and then density-certify fails (TIP3:90@30Å → ~0.1 g/cm³).
LB_ARGS=(
  --output-dir "$LIQUID_DIR"
  --seed "$SEED"
  --target-density-g-cm3 "$TARGET_DENSITY"
)
if [[ "$BOX_MODE" == "count" ]]; then
  echo "  mode=count: scale TIP3:1 to ρ=${TARGET_DENSITY} in L=${BOX_SIZE} Å"
  LB_ARGS+=(
    --composition "TIP3:1"
    --box-auto count
    --box-size "$BOX_SIZE"
  )
else
  echo "  mode=density: TIP3:${N_MOL} → L from ρ=${TARGET_DENSITY} g/cm³"
  LB_ARGS+=(
    --composition "TIP3:${N_MOL}"
    --box-auto density
  )
fi

mmml liquid-box "${LB_ARGS[@]}"

if [[ ! -f "$LIQUID_DIR/box.json" ]]; then
  echo "FAILED: missing $LIQUID_DIR/box.json" >&2
  exit 1
fi

# Surface liquid-box status (box.json is written even on fail).
uv run python - <<PY
import json
from pathlib import Path
p = Path("$LIQUID_DIR/box.json")
d = json.loads(p.read_text())
status = str(d.get("status", "?"))
side = d.get("box_side_A", d.get("final_cubic_side_A"))
rho = d.get("density_g_cm3")
msg = d.get("message") or ""
print(f"liquid-box status={status}  L={side} Å  ρ={rho} g/cm³")
if msg:
    print(f"  message: {msg}")
if status != "pass":
    raise SystemExit(
        "liquid-box certification failed — see REPORT.md / box.json message "
        "(usually density gate: N and L must match --target-density-g-cm3)."
    )
PY

echo ""
echo "=== [2/2] pressure MC + 1D refine → box.json ==="
uv run python - <<PY
from pathlib import Path
from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
    BoxPressureOptConfig,
    run_box_pressure_opt_from_box_json,
)

cfg = BoxPressureOptConfig(
    target_pressure_atm=float("$TARGET_P_ATM"),
    temperature_K=float("$TEMP_K"),
    seed=int("$SEED"),
    run_1d_refine=True,
    run_cpt_refine=False,  # short CPT refine: enable when CHARMM CPT runner is wired
)
result = run_box_pressure_opt_from_box_json(
    Path("$LIQUID_DIR"),
    output_dir=Path("$OPT_DIR"),
    config=cfg,
)
print(result.message)
print(f"wrote {result.box_json_path}")
if result.box_json_path is None or not Path(result.box_json_path).is_file():
    raise SystemExit(1)
PY

echo "Pass: $OPT_DIR/box.json (final_cubic_side_A + pressure provenance)"
echo "Next: hybrid smoke from certified CRD/box.json (fixed L), then CHARMM CPT equi."
