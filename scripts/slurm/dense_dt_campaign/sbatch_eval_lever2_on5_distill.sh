#!/usr/bin/env bash
#SBATCH --job-name=ddc-eval-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-eval-distill-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-eval-distill-%j.err
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
cd "$ROOT"
source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
export LJ_DEVICE=gpu JAX_PLATFORMS=cuda MMML_MLPOT_DEVICE=gpu MMML_JAX_WARMUP_DEVICE=gpu
export UV_NO_SYNC=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "ROOT=$ROOT job=$SLURM_JOB_ID host=$(hostname) $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv || true
python - <<'PY'
import jax, sys
d=jax.devices()
print('JAX devices:', d)
sys.exit(0 if any('cuda' in str(x).lower() or 'gpu' in str(x).lower() for x in d) else 4)
PY

uv run python scripts/slurm/dense_dt_campaign/eval_lever2_on5_distill_sweep.py
echo "done $(date -Is)"
