#!/usr/bin/env bash
#SBATCH --job-name=ddc-on5-softwell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-softwell-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-on5-softwell-%j.err
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs artifacts/lj_scales/ckpts
echo "ROOT=$ROOT job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
bash scripts/slurm/dense_dt_campaign/train_lever2_on5_softwell.sh
