#!/usr/bin/env bash
#SBATCH --job-name=spooky-mbd-2d
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/diagnostics/spooky-mbd-2d-%j.out
#SBATCH --error=artifacts/diagnostics/spooky-mbd-2d-%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64 XLA_PYTHON_CLIENT_PREALLOCATE=false

OUT="$ROOT/results/dimer_scan_campaign/mbd_checkpoint_comparison"
MBD="$HOME/qcml_runs/mbd_restart_20260711-100037/epoch-0100"
STEP="$ROOT/artifacts/spooky_so3lr/step-00003000_params.json"
MUON="$ROOT/artifacts/spooky_so3lr_muon_tight/epoch-0001_params.json"
mkdir -p "$OUT" "$ROOT/artifacts/diagnostics"
cp "$ROOT/results/dimer_scan_campaign/full_comparison/scan_results_with_references.csv" "$OUT/scan_results.csv"

"$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint "$STEP" --spookynet-mbd-checkpoint "$MBD" \
  --spookynet-tag step3000_mbd --skip-xtb --output-dir "$OUT"
"$MMML_PYTHON" scripts/run_dimer_scan_campaign.py \
  --spookynet-checkpoint "$MUON" --spookynet-mbd-checkpoint "$MBD" \
  --spookynet-tag muon_epoch1_mbd --skip-xtb --output-dir "$OUT"

"$MMML_PYTHON" scripts/clean_dimer_scan_data.py \
  --input "$OUT/scan_results.csv" --output "$OUT/scan_results_clean.csv" \
  --audit "$OUT/rejected_points.csv"
"$MMML_PYTHON" scripts/plot_2d_pes.py \
  --csv "$OUT/scan_results_clean.csv" --output-dir "$OUT/pes_2d" --max-cols 4 \
  --backends spookynet_hybrid_hybrid_step3000 \
    spookynet_hybrid_step3000_mbd spookynet_hybrid_muon_epoch1_mbd \
    xtb_gfn2 ccsd_def2svp_gpu4pyscf_cp ccsd_def2svpd_gpu4pyscf_cp \
    mp2_def2svp_gpu4pyscf_cp hf_def2svp_gpu4pyscf_cp \
    pbe0_def2svp_gpu4pyscf_cp pbe0_def2svp_gpu4pyscf_d3bj_cp
