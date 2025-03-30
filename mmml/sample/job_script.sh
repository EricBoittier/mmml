#!/bin/bash
#SBATCH --job-name=process_traj
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=process_traj_%A_%a.out
#SBATCH --error=process_traj_%A_%a.err

# Load necessary modules (adjust as needed)
#module load python/3.9

# Run the processing script
python features.py \
    --logfile ${LOGFILE} \
    --psf ${PSF} \
    --dcd ${DCD} \
    --pdb ${PDB} \
    --start ${START} \
    --end ${END} \
    --stride 1 \
    --samples_per_frame ${SAMPLES_PER_FRAME} \
    --n_find ${N_FIND}
