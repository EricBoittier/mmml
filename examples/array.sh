#!/usr/bin/env bash
#SBATCH -J sweep_cutoffs
##SBATCH -A your_account
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -o sweep_%A_%a.out
#SBATCH -e sweep_%A_%a.err
#SBATCH --array=0-150   # adjust to number of valid combos
#SBATCH --exclude=gpu14

#set -euo pipefail

hostname


nvidia-smi


REPO_ROOT="/pchem-data/meuwly/boittier/home/mmml"
DATASET="$REPO_ROOT/mmml/data/fixed-acetone-only_MP2_21000.npz"
CHECKPOINT="$REPO_ROOT/mmml/physnetjax/ckpts/test-70821ae3-d06a-4c87-9a2b-f5889c298376"
OUTDIR="$REPO_ROOT/examples/cutoffs_array"
mkdir -p "$OUTDIR"

# Define grids
MLS=(0.01 0.1 1.5 2.0 2.5)
SWONS=(0.01 3.0 2.0 4.0 5.0 6.0 7.0)
MCUTS=(0.01 5.0 5.5 6.0 6.5 7.0 7.5)

# Build the valid combo list deterministically
mapfile -t COMBOS < <(
  for ml in "${MLS[@]}"; do
    for sw in "${SWONS[@]}"; do
      for mc in "${MCUTS[@]}"; do
        awk "BEGIN{if ($sw < $mc && ($mc-$sw)>=1.0) print \"$ml $sw $mc\"}"
      done
    done
  done
)

idx=${SLURM_ARRAY_TASK_ID}
if (( idx < 0 || idx >= ${#COMBOS[@]} )); then
  echo "Index $idx out of range (${#COMBOS[@]} combos)"; exit 1
fi

read ml sw mc <<<"${COMBOS[$idx]}"
tag="ml${ml}_sw${sw}_mc${mc}"
out="$OUTDIR/result_0_${tag}.json"
log="$OUTDIR/run_0_${tag}.log"

echo "${COMBOS[$idx]}"

exit

module purge
module load gcc
# module load cuda/12.6.1
# module load cudnn/9.4


echo $PWD

# Activate your venv and deps like in run.sh
source ~/mmml/.venv/bin/activate

python "demo_pdbfile.py"           --dataset "$DATASET"             --checkpoint "$CHECKPOINT"               --units eV                 --sample-index 0                   --n-monomers 20                     --atoms-per-monomer 10                       --ml-cutoff $ml                          --mm-switch-on $sw     --output-prefix "results"$mc.$ml.$sw                     --mm-cutoff $mc                             --include-mm                               --output $OUTDIR"/results"$mc.$ml.$sw".json"                                 --pdbfile "init-packmol.pdb"
