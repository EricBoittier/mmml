#!/usr/bin/env bash
#SBATCH --job-name=ddc-train-on5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-train-on5-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-train-on5-%j.err
#
# GPU fine-tune at mm_switch_on=5 (soft lever-2 / DDC_HANDOFF=soft).
#
# From repo root:
#   mkdir -p artifacts/lj_scales/dense_dt_campaign/logs
#   sbatch scripts/slurm/dense_dt_campaign/sbatch_train_lever2_on5.sh
#
# Or via the submit helper (records job id):
#   bash scripts/slurm/dense_dt_campaign/submit_train_lever2_on5.sh
#
# Optional env overrides: DDC_ON5_EPOCHS, DDC_ON5_TAG, DDC_ON5_CKPT, ...
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${MMML_ROOT:-}" && -d "${MMML_ROOT}" ]]; then
  ROOT="$(cd "${MMML_ROOT}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"
mkdir -p artifacts/lj_scales/dense_dt_campaign/logs artifacts/lj_scales/ckpts
echo "ROOT=$ROOT job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is)"
bash scripts/slurm/dense_dt_campaign/train_lever2_on5.sh
