#!/usr/bin/env bash
#SBATCH --job-name=spooky-s3000-2d
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/diagnostics/spooky-s3000-2d-%j.out
#SBATCH --error=artifacts/diagnostics/spooky-s3000-2d-%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"

export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ORBAX="$ROOT/artifacts/spooky_so3lr/step-00003000"
JSON="$ROOT/artifacts/spooky_so3lr/step-00003000_params.json"
OUT="$ROOT/results/dimer_scan_campaign/full_comparison"

mkdir -p "$ROOT/artifacts/diagnostics" "$OUT"
if [[ ! -s "$JSON" ]]; then
  "$MMML_PYTHON" -m mmml.cli.__main__ orbax-to-json "$ORBAX" -o "$JSON"
fi

"$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint "$JSON" \
  --spookynet-tag hybrid_step3000 \
  --skip-xtb \
  --output-dir "$OUT"

"$MMML_PYTHON" scripts/plot_2d_pes.py \
  --csv "$OUT/scan_results.csv" \
  --output-dir "$OUT/pes_2d"
