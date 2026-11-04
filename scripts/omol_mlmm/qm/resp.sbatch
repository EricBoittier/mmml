#!/bin/bash
#SBATCH -p short -c 8 --mem=12G -t 04:00:00 -x node08 -J resp -o /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/residues/qm/logs/resp.%A_%a.out
cd /mmhome/boittier/home/metatomic-runs/training/omol_universal/mlmm_ft/residues/qm
/mmhome/boittier/home/miniforge3/envs/openff-toolkit/bin/python resp.py ${JOBS:-jobs.json} ${SLURM_ARRAY_TASK_ID:-0} ${NT:-1} ${OUTD:-out}
