
#!/bin/bash

#SBATCH --job-name=GPU_JOB
#SBATCH --time=01:00:00
#SBATCH --qos=gpu6hours
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=1
#SBATCH --array=0-1983
#SBATCH --cpus-per-task=2
#SBATCH --partition=rtx4090 #a100  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc


#module load CUDA
source ~/mmml/.venv/bin/activate

X=$SLURM_ARRAY_TASK_ID
echo $X
#python ~/mmml/mmml/pyscf4gpuInterface/calcs.py --mol "/scicore/home/meuwly/boitti0000/data/wateresp/xyzs/water_"$X".xyz" --energy --gradient --dens_esp --xc PBE0 --basis 6-31g --output "output/"$X".pkl"

uv run bash 

