#!/bin/bash
#SBATCH --job-name=jaxmd-cgenff-spoof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/jaxmd_cgenff_spoof_smoke/slurm-%j.out
#SBATCH --error=artifacts/jaxmd_cgenff_spoof_smoke/slurm-%j.err

set -euo pipefail
cd /mmhome/boittier/home/karml_cursor

export JAX_ENABLE_X64=1
# Prefer GPU if present; fall back handled by JAX.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export KARML_PYTHON="${KARML_PYTHON:-/mmhome/boittier/home/karml/.venv/bin/python}"
export PYTHONPATH="/mmhome/boittier/home/karml_cursor${PYTHONPATH:+:$PYTHONPATH}"
export KARML_CKPT="${KARML_CKPT:-/mmhome/boittier/home/karml_cursor/examples/ckpts_json/DESdimers_params.json}"

JOB="${1:-}"
if [[ -n "$JOB" ]]; then
  bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all.sh "$JOB"
else
  bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all.sh
fi
