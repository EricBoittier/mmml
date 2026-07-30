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
export MMML_PYTHON="${MMML_PYTHON:-/mmhome/boittier/home/mmml/.venv/bin/python}"
export PYTHONPATH="/mmhome/boittier/home/mmml_cursor${PYTHONPATH:+:$PYTHONPATH}"
export MMML_ALLOW_SELECTIVE_BONDED_BLOCK=1

MODE="${1:-compare}"
if [[ "$MODE" == "native" || "$MODE" == "all" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS_NATIVE:-cuda,cpu}"
  bash workflows/jaxmd_cgenff_spoof_smoke/scripts/run_all_native.sh
fi
if [[ "$MODE" == "compare" || "$MODE" == "all" ]]; then
  # Monomer E/F parity: JAX on CPU, PyCHARMM OpenCL on the allocated GPU node.
  # Bonded-only by default (matches jax_mm_spoof); set COMPARE_INCLUDE_MM=1 for full MM.
  export JAX_PLATFORMS=cpu
  export PYTHONUNBUFFERED=1
  COMPARE_ARGS=()
  if [[ "${COMPARE_INCLUDE_MM:-0}" != "1" ]]; then
    COMPARE_ARGS+=(--no-mm)
  fi
  "$MMML_PYTHON" -u workflows/jaxmd_cgenff_spoof_smoke/scripts/compare_to_charmm.py "${COMPARE_ARGS[@]}"
  "$MMML_PYTHON" -u workflows/jaxmd_cgenff_spoof_smoke/scripts/report_charmm_compare.py || true
fi
