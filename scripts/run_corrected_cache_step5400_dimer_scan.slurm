#!/usr/bin/env bash
#SBATCH --job-name=spooky-v2-s4600-2d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/diagnostics/spooky-v2-s4600-2d-%j.out
#SBATCH --error=artifacts/diagnostics/spooky-v2-s4600-2d-%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64 XLA_PYTHON_CLIENT_PREALLOCATE=false

ORBAX="$ROOT/artifacts/spooky_so3lr/step-00004600"
JSON="$ROOT/artifacts/spooky_so3lr/step-00004600_params.json"
OUT="$ROOT/results/dimer_scan_campaign/corrected_cache_step4600"
PRIOR="$ROOT/results/dimer_scan_campaign/recent_checkpoint_series/scan_results.csv"
mkdir -p "$OUT" "$ROOT/artifacts/diagnostics"

[[ -s "$JSON" ]] || "$MMML_PYTHON" -m mmml.cli.__main__ orbax-to-json "$ORBAX" -o "$JSON"
cp "$PRIOR" "$OUT/scan_results.csv"

"$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint "$JSON" --spookynet-no-mbd \
  --spookynet-tag corrected_v2_step4600 --skip-xtb --output-dir "$OUT"

"$MMML_PYTHON" scripts/clean_dimer_scan_data.py \
  --input "$OUT/scan_results.csv" --output "$OUT/scan_results_clean.csv" \
  --audit "$OUT/rejected_points.csv"

"$MMML_PYTHON" scripts/plot_2d_pes.py \
  --csv "$OUT/scan_results_clean.csv" --output-dir "$OUT/pes_2d_shared" \
  --shared-vmin -10 --shared-vmax 20 --max-cols 4 \
  --backends spookynet_hybrid_corrected_v2_step4600 \
    spookynet_hybrid_step45600 spookynet_hybrid_step44600 \
    mp2_def2svp_gpu4pyscf_cp hf_def2svp_gpu4pyscf_cp \
    pbe0_def2svp_gpu4pyscf_cp pbe0_def2svp_gpu4pyscf_d3bj_cp \
    ccsd_def2svp_gpu4pyscf_cp ccsd_def2svpd_gpu4pyscf_cp

"$MMML_PYTHON" scripts/plot_2d_pes.py \
  --csv "$OUT/scan_results_clean.csv" --output-dir "$OUT/pes_2d_failure_full" \
  --keep-energy-outliers --max-cols 4 \
  --backends spookynet_hybrid_corrected_v2_step4600 \
    spookynet_hybrid_step45600 spookynet_hybrid_step44600 \
    mp2_def2svp_gpu4pyscf_cp pbe0_def2svp_gpu4pyscf_cp \
    ccsd_def2svp_gpu4pyscf_cp
