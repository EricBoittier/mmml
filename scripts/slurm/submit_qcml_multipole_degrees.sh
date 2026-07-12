#!/usr/bin/env bash
set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
cd "$ROOT"
mkdir -p logs

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
DEGREES="${DEGREES:-1 2 3}"

base_export_vars=(
  "ALL"
  "RUN_TAG=$RUN_TAG"
  "MMML_ROOT=${MMML_ROOT:-$HOME/mmml}"
  "MMML_PYTHON=${MMML_PYTHON:-$HOME/mmml/.venv/bin/python}"
  "CACHE=${CACHE:-$HOME/orbax_cache/qcml_multipoles_traceless}"
  "MAX_STRUCTURES=${MAX_STRUCTURES:-}"
  "MAX_ATOMS=${MAX_ATOMS:-32}"
  "BATCH_SIZE=${BATCH_SIZE:-8}"
  "BUCKET_WIDTH=${BUCKET_WIDTH:-4}"
  "EPOCHS=${EPOCHS:-100}"
  "SAVE_EVERY=${SAVE_EVERY:-1}"
  "SAVE_OPT_STATE=${SAVE_OPT_STATE:-0}"
  "LEARNING_RATE=${LEARNING_RATE:-1e-4}"
  "WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}"
  "GRADIENT_CLIP_NORM=${GRADIENT_CLIP_NORM:-1.0}"
  "HUBER_DELTA=${HUBER_DELTA:-1.0}"
  "TARGET_SCALE_MODE=${TARGET_SCALE_MODE:-q95}"
  "OUTLIER_QUANTILE=${OUTLIER_QUANTILE:-0.95}"
  "OUTLIER_DEGREE_MODE=${OUTLIER_DEGREE_MODE:-component}"
  "COMPOSE_DIPOLE_FROM_ATOMIC=${COMPOSE_DIPOLE_FROM_ATOMIC:-1}"
  "ENFORCE_TOTAL_CHARGE=${ENFORCE_TOTAL_CHARGE:-1}"
  "VALIDATION_SHARDS=${VALIDATION_SHARDS:-2}"
  "TEST_SHARDS=${TEST_SHARDS:-2}"
  "EXCLUDE_NEWEST=${EXCLUDE_NEWEST:-1}"
  "DISABLE_PINNED_HOST_TRANSFER=${DISABLE_PINNED_HOST_TRANSFER:-0}"
  "FEATURES=${FEATURES:-64}"
  "NUM_ITERATIONS=${NUM_ITERATIONS:-3}"
  "NUM_BASIS_FUNCTIONS=${NUM_BASIS_FUNCTIONS:-16}"
  "CUTOFF=${CUTOFF:-6.0}"
)

jobs=()
for degree in $DEGREES; do
  case "$degree" in
    1|2|3) ;;
    *) echo "DEGREES must contain only 1, 2, and/or 3; got $degree" >&2; exit 2 ;;
  esac
  workdir_var="WORKDIR=${WORKDIR_BASE:-$HOME/qcml_runs}/multipoles_l${degree}_${RUN_TAG}"
  export_vars=("${base_export_vars[@]}" "TARGET_DEGREE=$degree" "$workdir_var")
  export_arg="$(IFS=,; echo "${export_vars[*]}")"
  job="$(sbatch --parsable --export="$export_arg" scripts/slurm/train_qcml_multipoles_degree.sh)"
  jobs+=("$job")
  echo "Submitted l=$degree multipole job: $job"
  echo "  workdir: ${WORKDIR_BASE:-$HOME/qcml_runs}/multipoles_l${degree}_${RUN_TAG}"
  echo "  log:     logs/qcml-multipoles-l${job}.out"
done

job_csv="$(IFS=,; echo "${jobs[*]}")"
echo "RUN_TAG:      $RUN_TAG"
echo "Monitor with: squeue -j $job_csv"
echo "Tail logs:"
for job in "${jobs[@]}"; do
  echo "  tail -F logs/qcml-multipoles-l${job}.out"
done
