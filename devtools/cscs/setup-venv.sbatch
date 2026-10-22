#!/bin/bash -l
# Step 2 only (image already built): thin venv + libcharmm/packmol + GPU smoke.
#SBATCH --job-name=mmml-venv
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=/users/boittier/mmml/devtools/cscs/build-%j.out
#SBATCH --error=/users/boittier/mmml/devtools/cscs/build-%j.out

set -euo pipefail
srun --environment=mmml bash /users/boittier/mmml/devtools/cscs/setup-venv.sh
