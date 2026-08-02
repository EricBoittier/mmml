#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=4000

module load gcc/gcc-12.2.0-cmake-3.25.1-openmpi-4.1.4
module load orca/orca-openmpi-6.1.0

/cluster/software/orca/orca_6_1_0_linux_x86-64_openmpi418/orca engrad.inp > engrad.out
