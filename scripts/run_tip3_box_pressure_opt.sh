#!/usr/bin/env bash
# TIP3 CHARMM-default box pressure opt prep (pinned liquid recipe):
#   liquid-box (MM certify) → live CHARMM PRSI MC/1D (+ CPT refine) → handoff
#   box_pressure_opt/{box.json,model.psf,model.crd}
#
# UNVERIFIED historical gpu09 recipe [evidence: tip3_count30_recipe]:
#   BOX_MODE=count  BOX_SIZE=30  TARGET_DENSITY=1.0  →  N≈903, L=30 Å, ρ≈1.00
#   MM GRMS ~0.04 kcal/mol/Å; worst inter-monomer ~1.17 Å (above prep floor).
#   Trust box.json status=pass over OpenMPI/PRRTE process exit codes.
#
# Default on gpu09: USE_CHARMM_PRESSURE=1 (live virial PRSI + CPT refine).
# Offline/CI: USE_CHARMM_PRESSURE=0 → synthetic P∝1/L³ (no libcharmm).
#
# Usage:
#   ./scripts/run_tip3_box_pressure_opt.sh
#   WIPE=0 ./scripts/run_tip3_box_pressure_opt.sh   # reuse existing liquid_box/
#   USE_CHARMM_PRESSURE=0 ./scripts/run_tip3_box_pressure_opt.sh  # synthetic P
#   BOX_MODE=density N_MOL=90 ./scripts/run_tip3_box_pressure_opt.sh  # L≈14 Å alt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- pinned defaults (do not change without re-validating liquid-box) --------
OUT_DIR="${OUT_DIR:-./scratch/tip3_physnet_ewald_ir/tip3_30A_box_opt}"
BOX_SIZE="${BOX_SIZE:-30}"
BOX_MODE="${BOX_MODE:-count}"          # pinned: count @ 30 Å → ~903 TIP3
TARGET_DENSITY="${TARGET_DENSITY:-1.0}"
TARGET_P_ATM="${TARGET_P_ATM:-1.0}"
TEMP_K="${TEMP_K:-300}"
SEED="${SEED:-42}"
# density-mode override only (ignored for BOX_MODE=count)
N_MOL="${N_MOL:-90}"
# Wipe rebuilds Packmol; set WIPE=0 to continue from a certified liquid_box/
WIPE="${WIPE:-1}"
# Live CHARMM path (gpu09 default). Set 0 for offline synthetic pressure.
USE_CHARMM_PRESSURE="${USE_CHARMM_PRESSURE:-1}"
RUN_CPT_REFINE="${RUN_CPT_REFINE:-1}"
MC_STEPS="${MC_STEPS:-32}"
CPT_NSTEP="${CPT_NSTEP:-500}"

mkdir -p "$OUT_DIR"
LIQUID_DIR="$OUT_DIR/liquid_box"
OPT_DIR="$OUT_DIR/box_pressure_opt"

if [[ "$WIPE" == "1" ]]; then
  rm -rf "$LIQUID_DIR" "$OPT_DIR"
fi
mkdir -p "$LIQUID_DIR"

echo "== TIP3 box pressure opt (CHARMM-default NpT prep) =="
echo "  pinned: BOX_MODE=count BOX_SIZE=30 ρ=1.0 → N≈903 @ L=30 Å"
echo "  liquid-box → $LIQUID_DIR"
echo "  pressure opt → $OPT_DIR"
echo "  target P = ${TARGET_P_ATM} atm   T = ${TEMP_K} K   ρ_target = ${TARGET_DENSITY} g/cm³"
echo "  BOX_MODE=$BOX_MODE  WIPE=$WIPE"

echo ""
echo "=== [1/2] mmml liquid-box (MM certify) ==="
LB_ARGS=(
  --output-dir "$LIQUID_DIR"
  --seed "$SEED"
  --target-density-g-cm3 "$TARGET_DENSITY"
  --rebuild-packmol
)
if [[ "$BOX_MODE" == "density" ]]; then
  echo "  mode=density: TIP3:${N_MOL} → L from ρ=${TARGET_DENSITY} g/cm³"
  LB_ARGS+=(
    --composition "TIP3:${N_MOL}"
    --box-auto density
  )
else
  echo "  mode=count (pinned): TIP3:1 → ρ=${TARGET_DENSITY} in L=${BOX_SIZE} Å"
  LB_ARGS+=(
    --composition "TIP3:1"
    --box-auto count
    --box-size "$BOX_SIZE"
  )
fi

set +e
mmml liquid-box "${LB_ARGS[@]}"
lb_rc=$?
set -e

if [[ ! -f "$LIQUID_DIR/box.json" ]]; then
  echo "FAILED: missing $LIQUID_DIR/box.json (liquid-box exit $lb_rc)" >&2
  exit 1
fi

uv run python - <<PY
import json
from pathlib import Path

p = Path("$LIQUID_DIR/box.json")
d = json.loads(p.read_text())
status = str(d.get("status", "?"))
side = float(d.get("box_side_A", d.get("final_cubic_side_A")))
rho = float(d.get("density_g_cm3") or 0.0)
n_mol = int(d.get("n_molecules") or 0)
msg = d.get("message") or ""
print(f"liquid-box status={status}  N={n_mol}  L={side:.4f} Å  ρ={rho:.6f} g/cm³")
if msg:
    print(f"  message: {msg}")
if status != "pass":
    raise SystemExit(
        "liquid-box certification failed — see REPORT.md / box.json message."
    )
# UNVERIFIED historical acceptance band [evidence: tip3_count30_recipe].
if str("$BOX_MODE") == "count":
    if abs(side - float("$BOX_SIZE")) > 0.05:
        raise SystemExit(f"pinned recipe expects L≈{float('$BOX_SIZE'):.1f} Å, got {side}")
    if abs(rho - float("$TARGET_DENSITY")) > 0.05:
        raise SystemExit(
            f"pinned recipe expects ρ≈{float('$TARGET_DENSITY'):.2f} g/cm³, got {rho}"
        )
    if n_mol < 800:
        raise SystemExit(f"pinned recipe expects N≳800 waters @ 30 Å, got {n_mol}")
lb_rc = int("$lb_rc")
if lb_rc != 0:
    print(
        f"note: liquid-box process exit={lb_rc} but box.json status=pass; continuing "
        "(OpenMPI/PRRTE often returns 1 after success)",
        flush=True,
    )
PY

echo ""
if [[ "$USE_CHARMM_PRESSURE" == "1" ]]; then
  echo "=== [2/2] live CHARMM PRSI MC/1D + CPT refine → handoff CRD/box.json ==="
else
  echo "=== [2/2] synthetic pressure MC/1D → box.json (offline) ==="
fi
uv run python - <<PY
from pathlib import Path
from mmml.interfaces.pycharmmInterface.mlpot.box_pressure_opt import (
    BoxPressureOptConfig,
    run_box_pressure_opt_from_box_json,
)

use_charmm = str("$USE_CHARMM_PRESSURE") == "1"
cfg = BoxPressureOptConfig(
    target_pressure_atm=float("$TARGET_P_ATM"),
    temperature_K=float("$TEMP_K"),
    seed=int("$SEED"),
    mc_steps=int("$MC_STEPS"),
    run_1d_refine=True,
    run_cpt_refine=(use_charmm and str("$RUN_CPT_REFINE") == "1"),
    cpt_nstep=int("$CPT_NSTEP"),
)
result = run_box_pressure_opt_from_box_json(
    Path("$LIQUID_DIR"),
    output_dir=Path("$OPT_DIR"),
    config=cfg,
    use_charmm_pressure=use_charmm,
)
print(result.message)
print(f"pressure_source={result.pressure_source}")
print(f"wrote {result.box_json_path}")
arts = result.artifacts or {}
if arts:
    print(f"artifacts: psf={arts.get('model_psf')} crd={arts.get('model_crd')}")
if result.box_json_path is None or not Path(result.box_json_path).is_file():
    raise SystemExit(1)
if use_charmm:
    crd = Path("$OPT_DIR") / "model.crd"
    psf = Path("$OPT_DIR") / "model.psf"
    if not crd.is_file() or not psf.is_file():
        raise SystemExit(f"live CHARMM path must write {psf} + {crd}")
PY

echo "Pass: $OPT_DIR/box.json (final_cubic_side_A + pressure provenance)"
if [[ "$USE_CHARMM_PRESSURE" == "1" ]]; then
  echo "Handoff: $OPT_DIR/{model.psf,model.crd,box.json} → STAGE=smoke at fixed L"
fi
echo "Next: STAGE=npt (CHARMM CPT) or STAGE=smoke (hybrid heat/NVE at fixed L)."
