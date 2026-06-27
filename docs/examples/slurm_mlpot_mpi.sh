#!/usr/bin/env bash
# Slurm example: Tier 1 (np=1) and Tier 2 (spatial ML) PyCHARMM MLpot jobs.
#
# Copy and edit paths before submitting:
#   sbatch docs/examples/slurm_mlpot_mpi.sh
#
# See docs/pycharmm-mpi.md for Phase 0–2 design.

#SBATCH --job-name=mmml-mlpot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=mmml-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# shellcheck source=../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"

export CHARMM_LIB_DIR="${CHARMM_LIB_DIR:-$HOME/.cache/mmml-charmm-build/tier_12000000_nodomdec/lib}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MMML_NO_JAX_COMPILE_THREADS="${MMML_NO_JAX_COMPILE_THREADS:-1}"

# --- Tier 1 (default): single MPI rank, one GPU ---------------------------
MMML_MPI_NP="${MMML_MPI_NP:-1}"
MPIRUN="$REPO_ROOT/scripts/mmml-charmm-mpirun.sh"

# --- Tier 2 (optional): spatial ML across ranks ---------------------------
# export MMML_MPI_NP=4
# export MMML_MLPOT_SPATIAL_MPI=1
# SPATIAL_FLAGS=(--ml-spatial-mpi --ml-gpu-count 1 --ml-batch-size 128)

SPATIAL_FLAGS=()

"$MPIRUN" md-system \
  --composition DCM:90 \
  --box-size 32 \
  --backend pycharmm \
  --md-stages mini,heat,equi \
  --checkpoint "${MMML_CKPT:-/path/to/DESdimers_params.json}" \
  --output-dir "artifacts/slurm_${SLURM_JOB_ID:-local}" \
  "${SPATIAL_FLAGS[@]}"
