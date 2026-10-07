#!/bin/bash
#SBATCH --job-name=meth-ewald-tiny
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/pbc_methane_ewald_smoke_tiny/slurm-%j.out
#SBATCH --error=artifacts/pbc_methane_ewald_smoke_tiny/slurm-%j.err

set -euo pipefail
cd /mmhome/boittier/home/karml_cursor
export KARML_PYTHON=/mmhome/boittier/home/karml/.venv/bin/python
export PYTHONPATH=/mmhome/boittier/home/karml_cursor
export JAX_ENABLE_X64=1
# Direct Python launch relies on CLI auto-rerun under mpirun; do not inherit a
# submitter shell that left KARML_NO_MPI_RERUN=1 from wrapper experiments.
unset KARML_NO_MPI_RERUN || true
export KARML_WORKFLOW_CONFIG=/mmhome/boittier/home/karml_cursor/workflows/pbc_methane_ewald/config.smoke.tiny.yaml
TAG="${1:?usage: sbatch submit_tiny_cell.sh TAG}"
bash workflows/pbc_methane_ewald/scripts/job_shell.sh "$TAG"
