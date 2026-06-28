#!/usr/bin/env bash
# End-to-end liquid DCM workflow on a GPU node (A100 / scicore-style).
#
# Phase 0 (optional): health-check --require-gpu --live
# Phase A: mmml liquid-box  — MM-only certification at bulk ρ (Packmol → MC → SD/ABNR)
# Phase B: mmml md-system   — hybrid MLpot mini (+ optional short heat) from certified box
#
# Recommended size (default N_DCM=40):
#   • 40 × DCM @ ρ=1.326 g/cm³ → L≈16.3 Å, 200 ML atoms
#   • Fits default CHARMM MLpot tier (max_Npr≈4M); JAX compiles in minutes on one A100
#   • Smaller / faster smoke: N_DCM=20 (L≈12.9 Å, 100 atoms)
#   • Larger (still OK on 4M tier): N_DCM=60 (L≈18.6 Å, 300 atoms)
#   • Avoid DCM:206+ without rebuilding xlarge lib (see docs/md-system-configs.md)
#
# Prerequisites:
#   module load GCC/14.2.0 OpenMPI/5.0.7-GCC-14.2.0 CMake/3.31.3-GCCcore-14.2.0
#   export OPENMPI_ROOT=$EBROOTOPENMPI
#   ./scripts/rebuild_charmm_mlpot.sh   # once; lib auto-discovered under setup/charmm
#   export MMML_CKPT=/path/to/checkpoint.json
#   uv sync --extra gpu
#
# Examples (any cwd):
#   export MMML_CKPT=~/mmml/mmml/models/physnetjax/defaults/hf_json/<ckpt>.json
#   ~/mmml/scripts/run_dcm_liquid_workflow.sh
#   N_DCM=20 ~/mmml/scripts/run_dcm_liquid_workflow.sh          # fastest liquid smoke
#   MD_STAGES=mini,heat,equi PS_HEAT=5 PS_EQUI=5 ./scripts/run_dcm_liquid_workflow.sh
#   SKIP_HEALTH=1 SKIP_LIQUID_BOX=1 ./scripts/run_dcm_liquid_workflow.sh  # md only
#   EXTRA_MD_ARGS=(--quiet) ~/mmml/scripts/run_dcm_liquid_workflow.sh
#
set -euo pipefail

MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"

# shellcheck source=scripts/resolve_mmml_env.sh
source "$MMML_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$MMML_ROOT"
PY="${MMML_PYTHON}"

# --- sizing (override with N_DCM=...) ---------------------------------------
N_DCM="${N_DCM:-40}"
DCM_RHO="${DCM_RHO:-1.326}"          # g/cm³, experimental bulk DCM ~298 K
LIQUID_BOX_PROFILE="${LIQUID_BOX_PROFILE:-standard}"  # standard | dense | conservative

# --- phases -----------------------------------------------------------------
SKIP_HEALTH="${SKIP_HEALTH:-0}"
SKIP_LIQUID_BOX="${SKIP_LIQUID_BOX:-0}"
SKIP_MD="${SKIP_MD:-0}"
REBUILD_BOX="${REBUILD_BOX:-0}"      # 1 = remove BOX_DIR before liquid-box

# --- md-system (Phase B) ----------------------------------------------------
MD_STAGES="${MD_STAGES:-mini,heat}"
MINI_NSTEP="${MINI_NSTEP:-50}"
PS_HEAT="${PS_HEAT:-3.0}"
PS_EQUI="${PS_EQUI:-0.0}"
ML_BATCH_SIZE="${ML_BATCH_SIZE:-64}"
ML_GPU_COUNT="${ML_GPU_COUNT:-1}"
TEMPERATURE="${TEMPERATURE:-300.0}"
PRESSURE="${PRESSURE:-1.0}"

# --- paths derived from N_DCM ------------------------------------------------
TAG="dcm${N_DCM}"
BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/$TAG}"
RUN_DIR="${RUN_DIR:-$TESTS_ROOT/runs/${TAG}_liquid}"
BOX_JSON="$BOX_DIR/box.json"
PSF="$BOX_DIR/model.psf"
CRD="$BOX_DIR/model.crd"

if [[ ! -d "$MMML_ROOT" ]]; then
  echo "MMML_ROOT not found: $MMML_ROOT" >&2
  exit 1
fi
if [[ ! -x "$MPIRUN" ]]; then
  echo "mpirun wrapper not found: $MPIRUN" >&2
  exit 1
fi

# shellcheck source=scripts/setup_jax_cuda_env.sh
source "$MMML_ROOT/scripts/setup_jax_cuda_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MMML_MPI_NP="${MMML_MPI_NP:-1}"

if [[ -z "${MMML_CKPT:-}" ]]; then
  DEFAULT_CKPT="$(
    find "$MMML_ROOT/mmml/models/physnetjax/defaults/hf_json" -maxdepth 1 -name '*_portable.json' 2>/dev/null | head -n 1
  )"
  if [[ -n "$DEFAULT_CKPT" ]]; then
    export MMML_CKPT="$DEFAULT_CKPT"
    echo "Using MMML_CKPT=$MMML_CKPT"
  else
    echo "Set MMML_CKPT to a PhysNet portable JSON checkpoint." >&2
    exit 1
  fi
fi

read -r EXPECTED_SIDE N_ATOMS <<EOF
$(
  "$PY" - <<PY
from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
    cubic_box_side_from_target_density,
    total_mass_g_for_composition,
)
n = int(${N_DCM})
comp = {"DCM": n}
mass = total_mass_g_for_composition(comp)
side = cubic_box_side_from_target_density(
    n_molecules=n,
    total_mass_g=mass,
    target_density_g_cm3=float(${DCM_RHO}),
)
print(f"{side:.2f} {5 * n}")
PY
)
EOF

echo "================================================================"
echo " DCM liquid workflow"
echo "================================================================"
echo " MMML_ROOT:      $MMML_ROOT"
echo " TESTS_ROOT:     $TESTS_ROOT"
echo " Composition:    DCM:${N_DCM} (${N_ATOMS} ML atoms)"
echo " Target rho:     ${DCM_RHO} g/cm³  (expected L≈${EXPECTED_SIDE} Å)"
echo " Box dir:        $BOX_DIR"
echo " Run dir:        $RUN_DIR"
echo " MD stages:      $MD_STAGES"
echo " Checkpoint:     $MMML_CKPT"
echo "================================================================"

if [[ "$SKIP_HEALTH" != "1" ]]; then
  echo "[phase 0] health-check (GPU + live MLpot DCM:2) ..."
  "$MPIRUN" health-check --require-gpu --live --checkpoint "$MMML_CKPT"
fi

if [[ "$SKIP_LIQUID_BOX" != "1" ]]; then
  if [[ "$REBUILD_BOX" == "1" && -d "$BOX_DIR" ]]; then
    echo "Removing existing box dir: $BOX_DIR"
    rm -rf "$BOX_DIR"
  fi
  if [[ -f "$BOX_JSON" ]]; then
    status="$("$PY" -c "import json; print(json.load(open('$BOX_JSON')).get('status','?'))")"
    echo "[phase A] reusing certified box (status=$status): $BOX_DIR"
  else
    echo "[phase A] liquid-box (MM only, profile=${LIQUID_BOX_PROFILE}) ..."
    mkdir -p "$(dirname "$BOX_DIR")"
    "$MPIRUN" liquid-box \
      --composition "DCM:${N_DCM}" \
      --target-density-g-cm3 "$DCM_RHO" \
      --profile "$LIQUID_BOX_PROFILE" \
      --output-dir "$BOX_DIR" \
      --charmm-sd-steps "${CHARMM_SD_STEPS:-100}" \
      --charmm-abnr-steps "${CHARMM_ABNR_STEPS:-200}" \
      --temperature "$TEMPERATURE" \
      --quiet
  fi
  if [[ ! -f "$PSF" || ! -f "$CRD" ]]; then
    echo "liquid-box did not write $PSF / $CRD" >&2
    exit 1
  fi
  if [[ -f "$BOX_JSON" ]]; then
    "$PY" - <<PY
import json, sys
p = "$BOX_JSON"
data = json.load(open(p))
print(f"  box.json: status={data.get('status')} L={data.get('box_side_A')} Å "
      f"rho={data.get('density_g_cm3')} g/cm³ worst_contact={data.get('worst_intermonomer_A')} Å")
if data.get("status") != "pass":
    print("  WARN: box certification status is not 'pass' — review $BOX_DIR/REPORT.md", file=sys.stderr)
PY
  fi
fi

if [[ "$SKIP_MD" != "1" ]]; then
  echo "[phase B] md-system (hybrid MLpot, from certified box) ..."
  mkdir -p "$RUN_DIR"
  MD_ARGS=(
    md-system
    --setup pbc_npt
    --backend pycharmm
    --composition "DCM:${N_DCM}"
    --from-psf "$PSF"
    --from-crd "$CRD"
    --skip-cluster-build
    --checkpoint "$MMML_CKPT"
    --output-dir "$RUN_DIR"
    --md-stages "$MD_STAGES"
    --mini-nstep "$MINI_NSTEP"
    --ps-heat "$PS_HEAT"
    --ps-equi "$PS_EQUI"
    --temperature "$TEMPERATURE"
    --pressure "$PRESSURE"
    --ml-batch-size "$ML_BATCH_SIZE"
    --ml-gpu-count "$ML_GPU_COUNT"
    --no-echeck
    --no-charmm-pre-minimize
    --max-grms-before-dyn 80.0
    --mm-switch-on 9.0
    --mm-switch-width 1.5
    --ml-switch-width 1.0
    --include-mm
    --seed 123
  )
  "$MPIRUN" "${MD_ARGS[@]}" "$@"
fi

echo "================================================================"
echo " Done."
echo "  Box:  $BOX_DIR/REPORT.md"
echo "  MD:   $RUN_DIR"
echo "================================================================"
