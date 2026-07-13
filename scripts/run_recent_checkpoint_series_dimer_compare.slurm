#!/usr/bin/env bash
#SBATCH --job-name=spooky-recent-2d
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/diagnostics/spooky-recent-2d-%j.out
#SBATCH --error=artifacts/diagnostics/spooky-recent-2d-%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64 XLA_PYTHON_CLIENT_PREALLOCATE=false

MBD="$HOME/qcml_runs/mbd_restart_20260711-100037/epoch-0100"
PRIOR="$ROOT/results/dimer_scan_campaign/mbd_checkpoint_comparison_step36800/scan_results.csv"
OUT="$ROOT/results/dimer_scan_campaign/recent_checkpoint_series"
mkdir -p "$OUT" "$ROOT/artifacts/diagnostics"
cp "$PRIOR" "$OUT/scan_results.csv"

for step in 42600 44600 45600; do
  padded=$(printf '%08d' "$step")
  orbax="$ROOT/artifacts/spooky_so3lr/step-$padded"
  json="$ROOT/artifacts/spooky_so3lr/step-${padded}_params.json"
  [[ -s "$json" ]] || "$MMML_PYTHON" -m mmml.cli.__main__ orbax-to-json "$orbax" -o "$json"
  "$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
    --spookynet-checkpoint "$json" --spookynet-no-mbd \
    --spookynet-tag "step${step}" --skip-xtb --output-dir "$OUT"
done

"$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint "$ROOT/artifacts/spooky_so3lr/step-00044600_params.json" \
  --spookynet-mbd-checkpoint "$MBD" --spookynet-tag step44600_mbd \
  --skip-xtb --output-dir "$OUT"

"$MMML_PYTHON" scripts/clean_dimer_scan_data.py \
  --input "$OUT/scan_results.csv" --output "$OUT/scan_results_clean.csv" \
  --audit "$OUT/rejected_points.csv"
"$MMML_PYTHON" scripts/plot_2d_pes.py \
  --csv "$OUT/scan_results_clean.csv" --output-dir "$OUT/pes_2d" --max-cols 4 \
  --backends spookynet_hybrid_step42600 spookynet_hybrid_step44600 \
    spookynet_hybrid_step44600_mbd spookynet_hybrid_step45600 \
    spookynet_hybrid_step36800 spookynet_hybrid_step19800 \
    mp2_def2svp_gpu4pyscf_cp hf_def2svp_gpu4pyscf_cp \
    pbe0_def2svp_gpu4pyscf_cp pbe0_def2svp_gpu4pyscf_d3bj_cp \
    ccsd_def2svp_gpu4pyscf_cp ccsd_def2svpd_gpu4pyscf_cp
