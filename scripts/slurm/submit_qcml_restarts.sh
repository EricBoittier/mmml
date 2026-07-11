#!/usr/bin/env bash
set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
cd "$ROOT"
mkdir -p logs

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export RUN_TAG

export_vars=(
  "ALL"
  "RUN_TAG=$RUN_TAG"
  "MMML_ROOT=${MMML_ROOT:-$HOME/mmml}"
  "MMML_PYTHON=${MMML_PYTHON:-$HOME/mmml/.venv/bin/python}"
  "MAX_STRUCTURES=${MAX_STRUCTURES:-}"
  "MAX_ATOMS=${MAX_ATOMS:-32}"
  "BATCH_SIZE=${BATCH_SIZE:-8}"
  "BUCKET_WIDTH=${BUCKET_WIDTH:-4}"
  "EPOCHS=${EPOCHS:-100}"
  "SAVE_EVERY=${SAVE_EVERY:-5}"
  "SAVE_OPT_STATE=${SAVE_OPT_STATE:-0}"
  "LEARNING_RATE=${LEARNING_RATE:-1e-4}"
  "WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}"
  "GRADIENT_CLIP_NORM=${GRADIENT_CLIP_NORM:-1.0}"
  "CHARGE_WEIGHT=${CHARGE_WEIGHT:-1.0}"
  "HUBER_DELTA=${HUBER_DELTA:-1.0}"
  "DEGREE_WEIGHTS=${DEGREE_WEIGHTS:-0.25:1:2:2}"
  "TARGET_SCALE_MODE=${TARGET_SCALE_MODE:-q95}"
  "OUTLIER_QUANTILE=${OUTLIER_QUANTILE:-0.99}"
  "OUTLIER_DEGREE_MODE=${OUTLIER_DEGREE_MODE:-component}"
  "COMPOSE_DIPOLE_FROM_ATOMIC=${COMPOSE_DIPOLE_FROM_ATOMIC:-1}"
  "ENFORCE_TOTAL_CHARGE=${ENFORCE_TOTAL_CHARGE:-1}"
  "VALIDATION_SHARDS=${VALIDATION_SHARDS:-2}"
  "TEST_SHARDS=${TEST_SHARDS:-2}"
  "EXCLUDE_NEWEST=${EXCLUDE_NEWEST:-1}"
  "DISABLE_PINNED_HOST_TRANSFER=${DISABLE_PINNED_HOST_TRANSFER:-0}"
)
if [[ -n "${MBD_WORKDIR:-}" ]]; then
  export_vars+=("MBD_WORKDIR=$MBD_WORKDIR")
fi
if [[ -n "${MULTIPOLE_WORKDIR:-}" ]]; then
  export_vars+=("MULTIPOLE_WORKDIR=$MULTIPOLE_WORKDIR")
fi
if [[ -n "${WORKDIR:-}" ]]; then
  export_vars+=("WORKDIR=$WORKDIR")
fi
export_arg="$(IFS=,; echo "${export_vars[*]}")"

mbd_job="$(sbatch --parsable --export="$export_arg" scripts/slurm/train_qcml_mbd_restart.sh)"
multipole_job="$(sbatch --parsable --export="$export_arg" scripts/slurm/train_qcml_multipoles_restart.sh)"

echo "Submitted MBD restart:        $mbd_job"
echo "Submitted multipole restart:  $multipole_job"
echo "RUN_TAG:                      $RUN_TAG"
echo "MBD workdir:                  ${MBD_WORKDIR:-${WORKDIR:-$HOME/qcml_runs/mbd_restart_${RUN_TAG}}}"
echo "Multipole workdir:            ${MULTIPOLE_WORKDIR:-${WORKDIR:-$HOME/qcml_runs/multipoles_restart_${RUN_TAG}}}"
echo "Exported settings:            $export_arg"
echo "Monitor with:                 squeue -j ${mbd_job},${multipole_job}"
echo "Logs:"
echo "  tail -F logs/qcml-mbd-restart-${mbd_job}.out"
echo "  tail -F logs/qcml-multipoles-restart-${multipole_job}.out"
