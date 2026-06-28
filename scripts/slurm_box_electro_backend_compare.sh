#!/usr/bin/env bash
# GPU Slurm job: DCM:5 reference box — MM on/off, electrostatics methods, NVE backends.
#
# Submit:
#   sbatch ~/mmml/scripts/slurm_box_electro_backend_compare.sh
#
# Monitor:
#   tail -f ~/tests/runs/dcm5_l25_electro_compare_*/slurm-*.out
#   cat ~/tests/runs/dcm5_l25_electro_compare_*/comparison.tsv
#
#SBATCH --job-name=box-electro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mmhome/boittier/home/tests/runs/slurm-box-electro-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/mmml}"
RUN_TAG="slurm${SLURM_JOB_ID:-local}"
export RUN_TAG
export RUN_ROOT="${RUN_ROOT:-$HOME/tests/runs/dcm5_l25_electro_compare_${RUN_TAG}}"
export TESTS_ROOT="${TESTS_ROOT:-$HOME/tests}"
export MMML_CKPT="${MMML_CKPT:-$HOME/mmml_tutorial/acodcm/ckpts/dcm1-c137fb42-1f65-4748-880b-8f8184a20f70}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MMML_MPI_NP="${MMML_MPI_NP:-1}"
export SKIP_BOX_BUILD="${SKIP_BOX_BUILD:-1}"

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

{
  echo "=== box electro compare Slurm job ==="
  echo "start: $(date -Iseconds)"
  echo "host:  $(hostname)"
  echo "job:   ${SLURM_JOB_ID:-local}"
  echo "RUN_ROOT: $RUN_ROOT"
  echo "MMML_CKPT: $MMML_CKPT"
  nvidia-smi -L 2>/dev/null || true
  echo "====================================="
} | tee "$RUN_ROOT/job_header.txt"

exec bash "$REPO_ROOT/tests/functionality/box_electro_compare/run_compare.sh" \
  2>&1 | tee "$RUN_ROOT/run.log"
