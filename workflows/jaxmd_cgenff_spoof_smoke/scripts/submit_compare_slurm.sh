#!/bin/bash
#SBATCH --job-name=charmm-cmp-spoof
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare/slurm-%j.out
#SBATCH --error=artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare/slurm-%j.err

set -euo pipefail
cd /mmhome/boittier/home/mmml_cursor
mkdir -p artifacts/jaxmd_cgenff_spoof_smoke/charmm_compare

export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export MMML_PYTHON="${MMML_PYTHON:-/mmhome/boittier/home/mmml/.venv/bin/python}"
export PYTHONPATH="/mmhome/boittier/home/mmml_cursor${PYTHONPATH:+:$PYTHONPATH}"
export MMML_ALLOW_SELECTIVE_BONDED_BLOCK=1

MODE="${1:-compare}"
if [[ "$MODE" == "native" || "$MODE" == "all" ]]; then
  bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all_native.sh
fi
if [[ "$MODE" == "compare" || "$MODE" == "all" ]]; then
  "$MMML_PYTHON" workflows/jaxmd_cgenff_spoof_smoke/scripts/compare_to_charmm.py
  "$MMML_PYTHON" workflows/jaxmd_cgenff_spoof_smoke/scripts/report_charmm_compare.py || true
fi
