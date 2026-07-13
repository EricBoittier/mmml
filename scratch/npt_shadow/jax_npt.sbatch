#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=jaxnpt
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mmhome/boittier/home/mmml/scratch/npt_shadow/jax_npt.slurm.log
source /mmhome/boittier/home/mmml/scratch/npt_shadow/env.sh
export OMP_NUM_THREADS=8 XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
cd /mmhome/boittier/home/mmml/scratch/npt_shadow
/mmhome/boittier/home/mmml/.venv/bin/python /mmhome/boittier/home/mmml/scratch/npt_shadow/npt_lj.py
