#!/bin/bash

# Path variables
TRUE_BASE_PATH="/pchem-data/meuwly/boittier/home/ressim/aceh"

# Loop over all subdirectories in the base path
for SUB_PATH in "${TRUE_BASE_PATH}"/sim*; do
    if [ -d "${SUB_PATH}" ]; then
        BASE_PATH="${SUB_PATH}"
       	echo $BASE_PATH 
        LOGFILE="${BASE_PATH}/log/equilibration_1_*.log"
        PSF="${BASE_PATH}/system.psf"
        DCD="${BASE_PATH}/dcd/equilibration_1_*.dcd"
        PDB="${BASE_PATH}/pdb/initial.pdb"

        # Job parameters
        BLOCK_SIZE=1000
        MAX_FRAMES=10000
        SAMPLES_PER_FRAME=1

        # Loop over n_find values
        for N_FIND in 1; do
            # Create job script
            cat << 'EOF' > job_script.sh
#!/bin/bash
#SBATCH --job-name=process_traj
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=process_traj_%A_%a.out
#SBATCH --error=process_traj_%A_%a.err

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
                
                sbatch \
                    --export=ALL,START=${start},END=${end},LOGFILE=${LOGFILE},PSF=${PSF},DCD=${DCD},PDB=${PDB},SAMPLES_PER_FRAME=${SAMPLES_PER_FRAME},N_FIND=${N_FIND} \
                    job_script.sh
                
                echo "Submitted job for frames ${start} to ${end}"
            
	done

        done
    
    fi


done
