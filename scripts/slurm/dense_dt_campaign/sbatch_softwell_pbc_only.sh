#!/usr/bin/env bash
#SBATCH --job-name=ddc-sw-pbc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-sw-pbc-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-sw-pbc-%j.err
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
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CKPT="${DDC_SW_CKPT:-artifacts/lj_scales/ckpts/params_hybrid_mm_lever2_on5_softwell_2026-08-02_22-15-54.json}"
SIDECAR="${DDC_SW_SIDECAR:-artifacts/lj_scales/ckpts/hybrid_mm_lever2_on5_softwell-657cb7db-74a1-4623-84a5-f772b8fe7928/hybrid_mm.json}"
OUT_PBC="${DDC_SW_PBC_OUT:-docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/pbc_translation.json}"

mkdir -p artifacts/lj_scales/dense_dt_campaign/logs "$(dirname "$OUT_PBC")"
echo "ROOT=$ROOT job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is)"

uv run python scripts/slurm/dense_dt_campaign/confirm_softwell_pbc.py \
  --checkpoint "$CKPT" \
  --sidecar "$SIDECAR" \
  --output "$OUT_PBC" \
  --box 24.0 \
  --n-monomers 120 \
  --atoms-per-monomer 5

python - <<PY
import json
from pathlib import Path
rep = json.loads(Path("$OUT_PBC").read_text())
print("pbc_ok:", rep.get("pbc_ok"))
print(json.dumps(rep.get("checks"), indent=2))
print(json.dumps(rep.get("cases"), indent=2))
if not rep.get("pbc_ok"):
    raise SystemExit(3)
PY
echo "=== softwell PBC done ==="
date -Is
