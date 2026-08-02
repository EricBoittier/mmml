#!/usr/bin/env bash
#SBATCH --job-name=ddc-sw-vis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu08,gpu09,gpu10
#SBATCH --output=artifacts/lj_scales/dense_dt_campaign/logs/ddc-sw-vis-%j.out
#SBATCH --error=artifacts/lj_scales/dense_dt_campaign/logs/ddc-sw-vis-%j.err
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
CSV="${DDC_SW_CSV:-artifacts/lj_scales/dense_dt_campaign/overbind_ablation/lever2_on5_softwell/distill_ep20_components.csv}"
OUT_POV="${DDC_SW_POV_OUT:-docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/povray}"
OUT_PBC="${DDC_SW_PBC_OUT:-docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/pbc_translation.json}"
POVRAY_BIN="${POVRAY:-/mmhome/boittier/home/miniforge3/envs/jaxphyscharmm/bin/povray}"

mkdir -p "$OUT_POV" artifacts/lj_scales/dense_dt_campaign/logs
echo "ROOT=$ROOT job=${SLURM_JOB_ID:-local} host=$(hostname) $(date -Is)"
echo "CKPT=$CKPT"
echo "SIDECAR=$SIDECAR"

# XY scatters (CPU-ok, fast; force CPU so login/CPU nodes without GPU work)
JAX_PLATFORMS=cpu uv run python scripts/slurm/dense_dt_campaign/make_softwell_deploy_figures.py

# POV stills of contact-ok soft geometries from softwell ep20 CSV
uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py \
  --checkpoint "$CKPT" \
  --sidecar "$SIDECAR" \
  --data artifacts/lj_scales/dataset_cgenff.npz \
  --components-csv "$CSV" \
  --out "$OUT_POV" \
  --mm-switch-on 5.0 \
  --ml-switch-width 1.5 \
  --mm-switch-width 5.0 \
  --r-grid 4.0 \
  --r-values "3.6,4.0,4.25,5.5" \
  --min-contact 2.0 \
  --n-directions 4 \
  --n-orientations 4 \
  --povray "$POVRAY_BIN"

# PBC translation confirmation on DCM:120 L=24
uv run python scripts/slurm/dense_dt_campaign/confirm_softwell_pbc.py \
  --checkpoint "$CKPT" \
  --sidecar "$SIDECAR" \
  --output "$OUT_PBC" \
  --box 24.0 \
  --n-monomers 120 \
  --atoms-per-monomer 5

echo "=== softwell deploy visuals done ==="
date -Is
ls -la docs/images/dense-dt-campaign/overbind_ablation/lever2_on5_softwell/ | head -40
