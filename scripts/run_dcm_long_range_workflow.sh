#!/usr/bin/env bash
# DCM liquid workflow with long-range Coulomb solver comparison (MIC / jax-pme / nvalchemiops / ScaFaCoS).
#
# Extends run_dcm_liquid_workflow.sh with solver sweeps and validation hooks.
#
# Phase 0: pytest + optional standalone long-range scripts (no PyCHARMM)
# Phase A: liquid-box MM certification (optional, same as DCM workflow)
# Phase B: md-system hybrid mini — one run per solver configuration
# Phase C: summary table of final energies / solver metadata
#
# Examples:
#   ~/mmml/scripts/run_dcm_long_range_workflow.sh
#   LR_SOLVERS=mic,jax_pme JAX_PME_METHODS=ewald,pme,p3m ~/mmml/scripts/run_dcm_long_range_workflow.sh
#   MM_NONBOND_MODE=periodic_external LR_SOLVERS=jax_pme,nvalchemiops_pme,scafacos SKIP_LIQUID_BOX=1 \
#     BOX_DIR=~/tests/boxes/dcm60_l32 ~/mmml/scripts/run_dcm_long_range_workflow.sh
#   SKIP_MD=1 ~/mmml/scripts/run_dcm_long_range_workflow.sh   # validation only
#
set -euo pipefail

MMML_ROOT="${MMML_ROOT:-$HOME/mmml}"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"

# shellcheck source=scripts/resolve_mmml_env.sh
source "$MMML_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$MMML_ROOT"
PY="${MMML_PYTHON}"

# --- inherit DCM sizing defaults (override as for liquid workflow) ------------
N_DCM="${N_DCM:-60}"
BOX_SIZE="${BOX_SIZE:-32}"
TAG="dcm${N_DCM}_l${BOX_SIZE}"
BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/$TAG}"
RUN_ROOT="${RUN_ROOT:-$TESTS_ROOT/runs/${TAG}_lr_solvers}"
PSF="$BOX_DIR/model.psf"
CRD="$BOX_DIR/model.crd"

# --- solver sweep -------------------------------------------------------------
# Comma-separated lists (no spaces).
LR_SOLVERS="${LR_SOLVERS:-mic,jax_pme}"
JAX_PME_METHODS="${JAX_PME_METHODS:-ewald,pme,p3m}"
SCAFACOS_METHODS="${SCAFACOS_METHODS:-ewald,p3m}"
MM_NONBOND_MODE="${MM_NONBOND_MODE:-jax_mic}"   # jax_mic | periodic_external
JAX_PME_SR_CUTOFF="${JAX_PME_SR_CUTOFF:-6.0}"
JAX_PME_DISPERSION="${JAX_PME_DISPERSION:-1}"   # 0 = Coulomb-only jax-pme LR

# --- phases -------------------------------------------------------------------
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
SKIP_LIQUID_BOX="${SKIP_LIQUID_BOX:-0}"
SKIP_MD="${SKIP_MD:-0}"
REBUILD_BOX="${REBUILD_BOX:-0}"

# --- md-system (short mini per solver) ----------------------------------------
MD_STAGES="${MD_STAGES:-mini}"
MINI_NSTEP="${MINI_NSTEP:-30}"
MMML_CKPT="${MMML_CKPT:-}"

if [[ ! -d "$MMML_ROOT" ]]; then
  echo "MMML_ROOT not found: $MMML_ROOT" >&2
  exit 1
fi

# shellcheck source=scripts/setup_jax_cuda_env.sh
source "$MMML_ROOT/scripts/setup_jax_cuda_env.sh" 2>/dev/null || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

IFS=',' read -r -a _lr_list <<< "${LR_SOLVERS// /}"
IFS=',' read -r -a _pme_list <<< "${JAX_PME_METHODS// /}"
IFS=',' read -r -a _scf_list <<< "${SCAFACOS_METHODS// /}"

echo "================================================================"
echo " DCM long-range solver workflow"
echo "================================================================"
echo " MMML_ROOT:        $MMML_ROOT"
echo " Box:              DCM:${N_DCM} L=${BOX_SIZE} Å → $BOX_DIR"
echo " Run root:         $RUN_ROOT"
echo " mm_nonbond_mode:  $MM_NONBOND_MODE"
echo " lr_solvers:       ${LR_SOLVERS}"
echo " jax_pme_methods:  ${JAX_PME_METHODS}"
echo " jax_pme_disp:     ${JAX_PME_DISPERSION}"
echo " scafacos_methods: ${SCAFACOS_METHODS}"
echo "================================================================"

have_scafacos() {
  "$PY" -c "from mmml.interfaces.scafacosInterface import have_scafacos; raise SystemExit(0 if have_scafacos() else 1)" 2>/dev/null
}

have_nvalchemiops_pme() {
  "$PY" -c "from mmml.interfaces.pycharmmInterface.long_range_backend import have_nvalchemiops_pme; raise SystemExit(0 if have_nvalchemiops_pme() else 1)" 2>/dev/null
}

if [[ "$SKIP_VALIDATION" != "1" ]]; then
  echo "[phase 0] long-range backend validation (pytest) ..."
  (cd "$MMML_ROOT" && JAX_PLATFORMS=cpu "$PY" -m pytest \
    tests/unit/test_jax_pme_lr_solver.py \
    tests/functionality/long_range/test_hybrid_jax_pme_mm.py \
    -q --tb=line)
  if have_scafacos; then
    echo "[phase 0] ScaFaCoS smoke (via mpirun wrapper) ..."
    export SCAFACOS_LIB="${SCAFACOS_LIB:-$HOME/.local/scafacos/lib/libfcs.so}"
    export LD_LIBRARY_PATH="${HOME}/.local/scafacos/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    (cd "$MMML_ROOT/tests/functionality/long_range" && \
      "$MPIRUN" "$PY" 04_scafacos_methods.py) || echo "  WARN: ScaFaCoS smoke failed (see above)"
  else
    echo "[phase 0] ScaFaCoS not installed — skipping 04_scafacos_methods.py"
  fi
fi

if [[ "$SKIP_LIQUID_BOX" != "1" ]]; then
  echo "[phase A] liquid-box (delegate to run_dcm_liquid_workflow.sh) ..."
  N_DCM="$N_DCM" BOX_SIZE="$BOX_SIZE" BOX_DIR="$BOX_DIR" \
    SKIP_HEALTH=1 SKIP_MD=1 REBUILD_BOX="$REBUILD_BOX" \
    "$MMML_ROOT/scripts/run_dcm_liquid_workflow.sh"
fi

if [[ -z "$MMML_CKPT" ]]; then
  MMML_CKPT="$(
    find "$MMML_ROOT/mmml/models/physnetjax/defaults/hf_json" -maxdepth 1 -name '*_portable.json' 2>/dev/null | head -n 1
  )"
fi
if [[ -z "$MMML_CKPT" && "$SKIP_MD" != "1" ]]; then
  echo "Set MMML_CKPT for hybrid md-system runs." >&2
  exit 1
fi

SUMMARY_TSV="$RUN_ROOT/solver_comparison.tsv"
mkdir -p "$RUN_ROOT"
echo -e "lr_solver\tjax_pme_method\tscafacos_method\tmm_nonbond_mode\trun_dir\tstatus\hybrid_grms_kcalmol_A" > "$SUMMARY_TSV"

read_hybrid_grms() {
  "$PY" -c "
from pathlib import Path
from mmml.interfaces.pycharmmInterface.lr_solver_grms_compare import read_hybrid_grms_from_output_dir
val = read_hybrid_grms_from_output_dir(Path('$1'))
print('' if val is None else f'{val:.6f}')
"
}

if [[ "$SKIP_MD" != "1" ]]; then
  if [[ ! -f "$PSF" || ! -f "$CRD" ]]; then
    echo "Missing certified box: $PSF / $CRD (run without SKIP_LIQUID_BOX=1)" >&2
    exit 1
  fi

  echo "[phase B] md-system solver sweep ..."
  for lr in "${_lr_list[@]}"; do
    lr="${lr// /}"
    [[ -z "$lr" ]] && continue

    if [[ "$lr" == "scafacos" ]] && ! have_scafacos; then
      echo "  SKIP lr_solver=scafacos (libfcs not found)"
      continue
    fi
    if [[ "$lr" == "nvalchemiops_pme" ]] && ! have_nvalchemiops_pme; then
      echo "  SKIP lr_solver=nvalchemiops_pme (nvalchemiops not found)"
      continue
    fi

    pme_methods=("")
    scf_methods=("")
    if [[ "$lr" == "jax_pme" ]]; then
      pme_methods=("${_pme_list[@]}")
    elif [[ "$lr" == "scafacos" ]]; then
      scf_methods=("${_scf_list[@]}")
    fi

    for jpm in "${pme_methods[@]:-""}"; do
      for scm in "${scf_methods[@]:-""}"; do
        jpm="${jpm// /}"
        scm="${scm// /}"
        tag="$lr"
        [[ -n "$jpm" ]] && tag="${tag}_${jpm}"
        [[ -n "$scm" ]] && tag="${tag}_${scm}"
        run_dir="$RUN_ROOT/${tag}_${MM_NONBOND_MODE}"
        echo "  → lr_solver=$lr jax_pme_method=${jpm:-—} scafacos_method=${scm:-—} → $run_dir"

        MD_ARGS=(
          md-system
          --config "$MMML_ROOT/mmml/cli/run/dcm_long_range_solvers.example.yaml"
          --setup pbc_npt
          --backend pycharmm
          --composition "DCM:${N_DCM}"
          --from-psf "$PSF"
          --from-crd "$CRD"
          --skip-cluster-build
          --checkpoint "$MMML_CKPT"
          --output-dir "$run_dir"
          --md-stages "$MD_STAGES"
          --mini-nstep "$MINI_NSTEP"
          --mm-nonbond-mode "$MM_NONBOND_MODE"
          --lr-solver "$lr"
          --jax-pme-sr-cutoff "$JAX_PME_SR_CUTOFF"
          --no-echeck
          --no-bonded-mm-mini
          --no-charmm-pre-minimize
          --max-grms-before-dyn 80.0
          --include-mm
          --seed 123
        )
        [[ -n "$jpm" ]] && MD_ARGS+=(--jax-pme-method "$jpm")
        if [[ "$lr" == "jax_pme" ]]; then
          if [[ "$JAX_PME_DISPERSION" == "0" || "$JAX_PME_DISPERSION" == "false" || "$JAX_PME_DISPERSION" == "False" ]]; then
            MD_ARGS+=(--no-jax-pme-dispersion)
          else
            MD_ARGS+=(--jax-pme-dispersion)
          fi
        fi
        [[ -n "$scm" ]] && MD_ARGS+=(--scafacos-method "$scm")

        status="ok"
        hybrid_grms=""
        if ! "$MPIRUN" "${MD_ARGS[@]}" "$@"; then
          status="fail"
        else
          hybrid_grms="$(read_hybrid_grms "$run_dir")"
        fi
        echo -e "${lr}\t${jpm}\t${scm}\t${MM_NONBOND_MODE}\t${run_dir}\t${status}\t${hybrid_grms}" >> "$SUMMARY_TSV"
      done
    done
  done
fi

echo "[phase C] summary: $SUMMARY_TSV"
column -t -s $'\t' "$SUMMARY_TSV" 2>/dev/null || cat "$SUMMARY_TSV"
if [[ "$SKIP_MD" != "1" && -f "$SUMMARY_TSV" ]]; then
  echo "[phase C] hybrid GRMS validation ..."
  if ! (cd "$MMML_ROOT/tests/functionality/long_range" && \
    JAX_PLATFORMS=cpu "$PY" 07_hybrid_grms_lr_solver_compare.py --summary-tsv "$SUMMARY_TSV"); then
    echo "WARN: hybrid GRMS cross-solver validation failed (see above)" >&2
  fi
fi
echo "Done. See docs/long-range-solver-tutorial.md"
