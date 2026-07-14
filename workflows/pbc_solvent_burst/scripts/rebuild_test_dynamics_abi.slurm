#!/usr/bin/env bash
#SBATCH --job-name=mmml-dynabi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/diagnostics/dynamics-abi-rebuild-%j.out
#SBATCH --error=artifacts/diagnostics/dynamics-abi-rebuild-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$HOME/mmml}"
cd "$REPO_ROOT"
mkdir -p artifacts/diagnostics

export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export OMP_NUM_THREADS=1

# Force a clean library so the test cannot accidentally load the pre-fix cache.
bash scripts/rebuild_charmm_mlpot.sh --clean --no-domdec --skip-packmol

source scripts/resolve_mmml_env.sh
mmml_resolve_env "$REPO_ROOT"
"$MMML_PYTHON" -m pytest tests/unit/test_charmm_dynamics_c_abi.py -q

# Reproduce the former dynamics_run_kw/dynopt crash with the fixed library,
# then continue through the matched PyCHARMM/JAX-MD NVE control.
export MMML_NVE_COMPARE_ROOT="$REPO_ROOT/artifacts/diagnostics/dcm10_nve_backend_compare_abi_fixed"
export MMML_REPO_ROOT="$REPO_ROOT"
bash workflows/pbc_solvent_burst/scripts/run_backend_nve_conservation.slurm
