#!/usr/bin/env bash
set -euo pipefail

ROOT="${MMML_ROOT:-$HOME/mmml}"
cd "$ROOT"
mkdir -p logs

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export RUN_TAG

mbd_job="$(sbatch --parsable scripts/slurm/train_qcml_mbd_restart.sh)"
multipole_job="$(sbatch --parsable scripts/slurm/train_qcml_multipoles_restart.sh)"

echo "Submitted MBD restart:        $mbd_job"
echo "Submitted multipole restart:  $multipole_job"
echo "RUN_TAG:                      $RUN_TAG"
echo "MBD workdir:                  ${MBD_WORKDIR:-${WORKDIR:-$HOME/qcml_runs/mbd_restart_${RUN_TAG}}}"
echo "Multipole workdir:            ${MULTIPOLE_WORKDIR:-${WORKDIR:-$HOME/qcml_runs/multipoles_restart_${RUN_TAG}}}"
echo "Monitor with:                 squeue -j ${mbd_job},${multipole_job}"
echo "Logs:"
echo "  tail -F logs/qcml-mbd-restart-${mbd_job}.out"
echo "  tail -F logs/qcml-multipoles-restart-${multipole_job}.out"
