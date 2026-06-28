#!/usr/bin/env bash
# Build reference DCM:5 box (if needed), run MM on/off + electrostatics + NVE backends.
#
# Usage:
#   export MMML_CKPT=~/mmml_tutorial/acodcm/ckpts/dcm1-...
#   bash tests/functionality/box_electro_compare/run_compare.sh
#
# Env:
#   SKIP_BOX_BUILD=1     reuse existing ~/tests/boxes/dcm5_l25_ref
#   SKIP_ENERGY=1        skip mini energy matrix
#   SKIP_NVE=1           skip NVE jobs
#   SKIP_WARMUP=1        skip serial JAX warmup (not recommended on GPU)
#   JOB_FILTER=energy_   run only jobs matching prefix
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMML_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
BOX_DIR="${BOX_DIR:-$TESTS_ROOT/boxes/dcm5_l25_ref}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$TESTS_ROOT/runs/dcm5_l25_electro_compare_${RUN_TAG}}"
CONFIG="$SCRIPT_DIR/config.yaml"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$MMML_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$MMML_ROOT"
PY="${MMML_PYTHON}"
MPIRUN="${MMML_MPIRUN_WRAPPER:-$MMML_ROOT/scripts/mmml-charmm-mpirun.sh}"

# shellcheck source=../../../scripts/setup_jax_cuda_env.sh
source "$MMML_ROOT/scripts/setup_jax_cuda_env.sh" 2>/dev/null || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export MMML_MPI_NP="${MMML_MPI_NP:-1}"

if [[ -z "${MMML_CKPT:-}" ]]; then
  MMML_CKPT="${MMML_CKPT:-$HOME/mmml_tutorial/acodcm/ckpts/dcm1-c137fb42-1f65-4748-880b-8f8184a20f70}"
fi
export MMML_CKPT

mkdir -p "$RUN_ROOT"
ln -sfn "$RUN_ROOT" "$SCRIPT_DIR/results"

echo "================================================================"
echo " Box electrostatics / backend compare"
echo " $(date -Iseconds)"
echo " MMML_ROOT:  $MMML_ROOT"
echo " BOX_DIR:    $BOX_DIR"
echo " RUN_ROOT:   $RUN_ROOT"
echo " MMML_CKPT:  $MMML_CKPT"
echo " JAX:        $JAX_PLATFORMS"
echo "================================================================"

if [[ "${SKIP_BOX_BUILD:-0}" != "1" ]]; then
  if [[ -f "$BOX_DIR/model.psf" && -f "$BOX_DIR/model.crd" ]]; then
    echo "[box] reuse certified box at $BOX_DIR"
  else
    echo "[box] building reference DCM:5 L=25 Å → $BOX_DIR"
    mkdir -p "$BOX_DIR"
    "$MPIRUN" liquid-box \
      --composition DCM:5 \
      --box-size 25 \
      --target-density-g-cm3 1.326 \
      --profile standard \
      -o "$BOX_DIR"
  fi
else
  echo "[box] SKIP_BOX_BUILD=1 — expect $BOX_DIR/model.psf"
fi

if [[ ! -f "$BOX_DIR/model.psf" || ! -f "$BOX_DIR/model.crd" ]]; then
  echo "Missing reference box: $BOX_DIR/model.psf / model.crd" >&2
  exit 1
fi

# Expand ~ in config paths for this run
export BOX_PSF="$BOX_DIR/model.psf"
export BOX_CRD="$BOX_DIR/model.crd"

FAIL=0

_warmup_mlpot_jax() {
  if [[ "${SKIP_WARMUP:-0}" == "1" ]]; then
    echo "[warmup] SKIP_WARMUP=1"
    return 0
  fi
  echo "[warmup] mmml warmup-mlpot-jax (serial, before mpirun) $(date -Iseconds)"
  # Slurm/srun exports PMI env; JAX/ptxas must not load OpenMPI libs during compile.
  while IFS= read -r _var; do
    [[ -n "$_var" ]] && unset "$_var" 2>/dev/null || true
  done < <(env | cut -d= -f1 | grep -E '^(OMPI_|PMI_|PMIX_|MPI_LOCALRANKID$|SLURM_MPI_TYPE$)' || true)
  export MMML_WARMUP_MLPOT_JAX_ONLY=1
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export MMML_JAX_COMPILE_THREADS="${MMML_JAX_COMPILE_THREADS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  if ! "$PY" -m mmml.cli.__main__ warmup-mlpot-jax \
      --checkpoint "$MMML_CKPT" \
      --n-monomers 5 \
      --box-side 25 \
      --spacing 5 \
      --ml-batch-size 64 \
      --ml-gpu-count 1 \
      --do-mm; then
    echo "ERROR: warmup-mlpot-jax failed" >&2
    return 1
  fi
  unset MMML_WARMUP_MLPOT_JAX_ONLY
}

if ! _warmup_mlpot_jax; then
  echo "FATAL: warmup-mlpot-jax failed; aborting compare run" >&2
  exit 1
fi

_run_job() {
  local job_id="$1"
  local out="$RUN_ROOT/$job_id"
  mkdir -p "$out"
  echo ""
  echo "[job] $job_id → $out  ($(date -Iseconds))"
  if ! "$PY" "$SCRIPT_DIR/run_one_job.py" \
      --config "$CONFIG" \
      --job-id "$job_id" \
      --output-dir "$out" \
      --from-psf "$BOX_PSF" \
      --from-crd "$BOX_CRD"; then
    echo "  FAILED: $job_id" >&2
    return 1
  fi
  return 0
}

ENERGY_JOBS=(
  energy_mic_mm
  energy_jax_pme_ewald_mm
  energy_jax_pme_pme_mm
  energy_jax_pme_p3m_mm
  energy_ml_only
)

NVE_JOBS=(
  nve_ase_mic_mm
  nve_ase_jax_pme_ewald_mm
  nve_ase_ml_only
  nve_jaxmd_mic_mm
  nve_jaxmd_jax_pme_ewald_mm
  nve_pycharmm_mic_mm
  nve_pycharmm_jax_pme_ewald_mm
)

_filter_job() {
  local id="$1"
  [[ -z "${JOB_FILTER:-}" ]] && return 0
  [[ "$id" == "$JOB_FILTER"* ]]
}

if [[ "${SKIP_ENERGY:-0}" != "1" ]]; then
  echo "[phase] energy (mini) jobs ..."
  for j in "${ENERGY_JOBS[@]}"; do
    _filter_job "$j" || continue
    _run_job "$j" || FAIL=1
  done
fi

if [[ "${SKIP_NVE:-0}" != "1" ]]; then
  echo "[phase] NVE backend jobs ..."
  for j in "${NVE_JOBS[@]}"; do
    _filter_job "$j" || continue
    _run_job "$j" || FAIL=1
  done
fi

echo "[collect] $(date -Iseconds)"
"$PY" "$SCRIPT_DIR/collect_results.py" \
  --run-root "$RUN_ROOT" \
  --config "$CONFIG" \
  --out-tsv "$RUN_ROOT/comparison.tsv" \
  --out-md "$RUN_ROOT/REPORT.md"

echo ""
echo "Done. RUN_ROOT=$RUN_ROOT"
echo "  comparison.tsv"
echo "  REPORT.md"
exit "$FAIL"
