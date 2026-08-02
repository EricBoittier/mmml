#!/usr/bin/env bash
#SBATCH --job-name=ddc-on5-sw-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-sw-eval-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-sw-eval-%j.err
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"
source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"
export PYTHONUNBUFFERED=1
export LJ_DEVICE=gpu
export JAX_PLATFORMS=cuda
export MMML_MLPOT_DEVICE=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p artifacts/lj_scales/dense_dt_campaign/logs
echo "ROOT=$ROOT job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is)"
uv run python scripts/slurm/dense_dt_campaign/eval_lever2_on5_softwell_sweep.py
