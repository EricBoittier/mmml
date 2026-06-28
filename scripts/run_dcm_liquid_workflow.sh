#!/usr/bin/env bash
# End-to-end liquid DCM workflow on a GPU node (A100 / scicore-style).
#
# Phase 0 (optional): health-check --require-gpu --live
# Phase A: mmml liquid-box  — MM-only certification (Packmol → MC → SD/ABNR → optional mini-NPT)
# Phase B: mmml md-system   — hybrid MLpot mini (+ optional heat/equi) from certified box
#
# MIC + liquid density (read this before changing N_DCM):
#   Bulk DCM @ ρ=1.326 g/cm³ gives L ≈ (N × 106 Å³)^(1/3).
#   MLpot PBC needs L/2 > cutnb (≈12–18 Å) → use L ≥ 28 Å (conservative: 32 Å).
#   At bulk ρ that means N ≳ 200 for L≥28 Å and N ≳ 310 for L≥32 Å — slow JAX / large tiers.
#
#   Practical compromise (default): DCM:60 in a fixed L=32 Å cube.
#     • MIC-safe (L/2 = 16 Å > typical cutnb)
#     • 300 ML atoms — fits default CHARMM tier (max_Npr≈4M)
#     • Initial placement is sub-bulk ρ; MC + NPT equi (dense profile / ps_equi) densify toward liquid
#   Documented in docs/md-system-configs.md (DCM:60 @ box_size 32).
#
#   Faster smoke (not production MIC): N_DCM=20 BOX_SIZE=45 (dilute, like long-range test configs)
#   True bulk ρ + MIC: BOX_AUTO=count BOX_SIZE=32 (→ ~DCM:308; slow; ensure_charmm_mlpot_limits)
#
# Prerequisites:
#   module load GCC/14.2.0 OpenMPI/5.0.7-GCC-14.2.0 CMake/3.31.3-GCCcore-14.2.0
#   export OPENMPI_ROOT=$EBROOTOPENMPI
#   ./scripts/rebuild_charmm_mlpot.sh
#   export MMML_CKPT=/path/to/checkpoint.json
#   uv sync --extra gpu
#
# Examples (any cwd):
#   export MMML_CKPT=~/mmml/mmml/models/physnetjax/defaults/hf_json/<ckpt>.json
#   ~/mmml/scripts/run_dcm_liquid_workflow.sh
#   N_DCM=90 BOX_SIZE=32 ~/mmml/scripts/run_dcm_liquid_workflow.sh
#   N_DCM=20 BOX_SIZE=45 SKIP_HEALTH=1 ~/mmml/scripts/run_dcm_liquid_workflow.sh  # fast dilute smoke
#   MD_STAGES=mini,equi PS_EQUI=10 ~/mmml/scripts/run_dcm_liquid_workflow.sh
#   SKIP_HEALTH=1 SKIP_LIQUID_BOX=1 BOX_DIR=~/tests/boxes/dcm60_l32 ~/mmml/scripts/run_dcm_liquid_workflow.sh
#   LIQUID_BOX_VERBOSE=1 ~/mmml/scripts/run_dcm_liquid_workflow.sh  # drop --quiet on liquid-box
#
# Note: CHARMM Fortran I/O fails on paths with uppercase letters (e.g. dcm60_L32).
# mmml stages I/O under $TMPDIR/mmml-charmm-io/ automatically; prefer lowercase box dirs.
#
set -euo pipefail

MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"

# shellcheck source=scripts/resolve_mmml_env.sh
source "$MMML_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$MMML_ROOT"
PY="${MMML_PYTHON}"

# --- sizing (override with N_DCM=..., BOX_SIZE=...) -------------------------
N_DCM="${N_DCM:-60}"
BOX_SIZE="${BOX_SIZE:-32}"              # Å; MIC-safe default (not bulk-ρ L for N=60)
DCM_RHO="${DCM_RHO:-1.326}"            # g/cm³ target for MC / NPT equilibration
MIC_MIN_BOX="${MIC_MIN_BOX:-28}"         # Å; warn below this (L/2 vs ml_cutoff/cutnb)
ML_CUTOFF="${ML_CUTOFF:-12.0}"
LIQUID_BOX_PROFILE="${LIQUID_BOX_PROFILE:-dense}"  # dense → liquid_prep + mini-NPT toward ρ
MINI_BOX_EQUIL_PS="${MINI_BOX_EQUIL_PS:-}"         # empty = liquid-box profile default (2 ps dense)

# --- phases -----------------------------------------------------------------
SKIP_HEALTH="${SKIP_HEALTH:-0}"
SKIP_LIQUID_BOX="${SKIP_LIQUID_BOX:-0}"
SKIP_MD="${SKIP_MD:-0}"
REBUILD_BOX="${REBUILD_BOX:-0}"

# --- md-system (Phase B) ----------------------------------------------------
MD_STAGES="${MD_STAGES:-mini,equi}"
MINI_NSTEP="${MINI_NSTEP:-50}"
PS_HEAT="${PS_HEAT:-3.0}"
PS_EQUI="${PS_EQUI:-10.0}"
ML_BATCH_SIZE="${ML_BATCH_SIZE:-64}"
ML_GPU_COUNT="${ML_GPU_COUNT:-1}"
TEMPERATURE="${TEMPERATURE:-300.0}"
PRESSURE="${PRESSURE:-1.0}"

# --- paths ------------------------------------------------------------------
TAG="dcm${N_DCM}_l${BOX_SIZE}"
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

read -r BULK_L N_ATOMS EFF_RHO MIC_OK <<EOF
$(
  "$PY" - <<PY
from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
    cubic_box_side_from_target_density,
    total_mass_g_for_composition,
)
n = int(${N_DCM})
L_bulk = float(${BOX_SIZE})
comp = {"DCM": n}
mass = total_mass_g_for_composition(comp)
L_rho = cubic_box_side_from_target_density(
    n_molecules=n,
    total_mass_g=mass,
    target_density_g_cm3=float(${DCM_RHO}),
)
# effective ρ if N molecules are placed in BOX_SIZE cube
import math
vol_a3 = float(${BOX_SIZE}) ** 3
mass_g = mass
vol_cm3 = vol_a3 * 1e-24
eff_rho = mass_g / vol_cm3 if vol_cm3 > 0 else 0.0
mic_min = float(${MIC_MIN_BOX})
ml_cut = float(${ML_CUTOFF})
mic_floor = max(mic_min, 2.0 * ml_cut + 2.0)
mic_ok = 1 if float(${BOX_SIZE}) >= mic_floor else 0
print(f"{L_rho:.2f} {5 * n} {eff_rho:.3f} {mic_ok}")
PY
)
EOF

echo "================================================================"
echo " DCM liquid workflow (MIC-aware)"
echo "================================================================"
echo " MMML_ROOT:      $MMML_ROOT"
echo " TESTS_ROOT:     $TESTS_ROOT"
echo " Composition:    DCM:${N_DCM} (${N_ATOMS} ML atoms)"
echo " Box side:       ${BOX_SIZE} Å  (bulk-ρ L would be ≈${BULK_L} Å only)"
echo " Effective ρ₀:   ${EFF_RHO} g/cm³ in ${BOX_SIZE} Å cube (target ${DCM_RHO} via MC/NPT)"
echo " MIC min L:      ${MIC_MIN_BOX} Å (2×ml_cutoff+2 ≈ $("$PY" -c "print(2*${ML_CUTOFF}+2)"))"
echo " Box dir:        $BOX_DIR"
echo " Run dir:        $RUN_DIR"
echo " MD stages:      $MD_STAGES"
echo " Checkpoint:     $MMML_CKPT"
echo "================================================================"

if [[ "$MIC_OK" != "1" ]]; then
  echo "ERROR: BOX_SIZE=${BOX_SIZE} Å is below MIC-safe minimum (~${MIC_MIN_BOX} Å)." >&2
  echo "  Use BOX_SIZE=32 (default) or N_DCM=20 BOX_SIZE=45 for dilute smoke." >&2
  exit 1
fi

if awk -v bulk="$BULK_L" -v box="$BOX_SIZE" 'BEGIN { exit !(bulk < box - 0.5) }'; then
  echo "NOTE: bulk-ρ cube (${BULK_L} Å) < BOX_SIZE (${BOX_SIZE} Å)."
  echo "      Starting sub-bulk; MC + NPT equi (${LIQUID_BOX_PROFILE} profile) work toward ρ=${DCM_RHO}."
fi

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
    echo "[phase A] liquid-box (MM only, L=${BOX_SIZE} Å, profile=${LIQUID_BOX_PROFILE}) ..."
    mkdir -p "$(dirname "$BOX_DIR")"
    LIQUID_BOX_ARGS=(
      liquid-box
      --composition "DCM:${N_DCM}"
      --box-size "$BOX_SIZE"
      --target-density-g-cm3 "$DCM_RHO"
      --profile "$LIQUID_BOX_PROFILE"
      --output-dir "$BOX_DIR"
      --charmm-sd-steps "${CHARMM_SD_STEPS:-100}"
      --charmm-abnr-steps "${CHARMM_ABNR_STEPS:-200}"
      --temperature "$TEMPERATURE"
    )
    if [[ -n "$MINI_BOX_EQUIL_PS" ]]; then
      LIQUID_BOX_ARGS+=(--mini-box-equil-ps "$MINI_BOX_EQUIL_PS")
    fi
    if [[ "${LIQUID_BOX_VERBOSE:-0}" != "1" ]]; then
      LIQUID_BOX_ARGS+=(--quiet)
    fi
    "$MPIRUN" "${LIQUID_BOX_ARGS[@]}"
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
print(
    f"  box.json: status={data.get('status')} L={data.get('box_side_A')} Å "
    f"rho={data.get('density_g_cm3')} g/cm³ "
    f"worst_contact={data.get('worst_intermonomer_A')} Å"
)
side = float(data.get("box_side_A") or 0)
if side > 0 and side < float(${MIC_MIN_BOX}) - 0.5:
    print(f"  WARN: certified L={side} Å < MIC_MIN_BOX=${MIC_MIN_BOX}", file=sys.stderr)
if data.get("status") != "pass":
    print("  WARN: box certification status is not 'pass' — review REPORT.md", file=sys.stderr)
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
    --no-bonded-mm-mini
    --no-charmm-pre-minimize
    --max-grms-before-dyn 80.0
    --mini-lattice-abnr-steps 0
    --density-prep-lattice-abnr-steps 0
    --mini-box-equil-ps 0
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
