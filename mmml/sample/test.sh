#!/bin/bash

# Path variables - adjust these to match your setup
BASE_PATH="/pchem-data/meuwly/boittier/home/ressim/aceh/sim_t_298.15_k_rho_1043.9_kgperm3_pNone_kPa"
LOGFILE="${BASE_PATH}/log/equilibration_1_20250317_203756.log"
PSF="${BASE_PATH}/system.psf"
DCD="${BASE_PATH}/dcd/equilibration_1_20250317_203756.dcd"
PDB="${BASE_PATH}/pdb/initial.pdb"

# Job parameters
BLOCK_SIZE=1000
MAX_FRAMES=10000  # Adjust this to your total number of frames
SAMPLES_PER_FRAME=1
N_FIND=3

# Create a temporary job script
cat << 'EOF' > job_script.sh
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
EOF

# Submit jobs for each block
for ((start=0; start<MAX_FRAMES; start+=BLOCK_SIZE)); do
    end=$((start + BLOCK_SIZE))
    
    # Submit the job with environment variables
    sbatch \
        --export=ALL,START=${start},END=${end},LOGFILE=${LOGFILE},PSF=${PSF},DCD=${DCD},PDB=${PDB},SAMPLES_PER_FRAME=${SAMPLES_PER_FRAME},N_FIND=${N_FIND} \
        job_script.sh
    
    echo "Submitted job for frames ${start} to ${end}"
    
    # Optional: add a small delay between submissions
    #sleep 1
done
