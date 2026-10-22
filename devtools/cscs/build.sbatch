#!/bin/bash -l
#SBATCH --job-name=mmml-build
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclude=nid007417        # no subuid range for boittier there (2026-10-02)
#SBATCH --output=/users/boittier/mmml/devtools/cscs/build-%j.out
#SBATCH --error=/users/boittier/mmml/devtools/cscs/build-%j.out

set -euo pipefail
srun bash /users/boittier/mmml/devtools/cscs/build.sh
srun --environment=mmml bash /users/boittier/mmml/devtools/cscs/setup-venv.sh
