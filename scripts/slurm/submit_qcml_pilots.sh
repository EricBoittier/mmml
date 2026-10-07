#!/usr/bin/env bash
set -euo pipefail

ROOT="${KARML_ROOT:-$HOME/karml}"
cd "$ROOT"
mkdir -p logs

mbd_job="$(sbatch --parsable scripts/slurm/train_qcml_mbd_pilot.sh)"
multipole_job="$(sbatch --parsable scripts/slurm/train_qcml_multipoles_pilot.sh)"

echo "Submitted MBD pilot:        $mbd_job"
echo "Submitted multipole pilot:  $multipole_job"
echo "Monitor with: squeue -j ${mbd_job},${multipole_job}"
