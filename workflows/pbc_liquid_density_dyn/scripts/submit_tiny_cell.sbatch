#!/bin/bash
#SBATCH --job-name=dcm-dens-tiny
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/pbc_liquid_density_dyn_smoke_tiny/slurm-%j.out
#SBATCH --error=artifacts/pbc_liquid_density_dyn_smoke_tiny/slurm-%j.err

set -euo pipefail
cd /mmhome/boittier/home/mmml_cursor
export MMML_PYTHON=/mmhome/boittier/home/mmml/.venv/bin/python
export PYTHONPATH=/mmhome/boittier/home/mmml_cursor
export JAX_ENABLE_X64=1
export MMML_CKPT=/mmhome/boittier/home/mmml_cursor/examples/ckpts_json/DESdimers_params.json
export MMML_WORKFLOW_CONFIG=/mmhome/boittier/home/mmml_cursor/workflows/pbc_liquid_density_dyn/config.smoke.tiny.gpu.yaml
export MMML_MPI_NP=1
# Direct Python launch (default) relies on CLI auto-rerun under mpirun.
unset MMML_NO_MPI_RERUN || true
TAG="${1:-dcm_8}"
bash workflows/pbc_liquid_density_dyn/scripts/job_shell.sh "$TAG"
