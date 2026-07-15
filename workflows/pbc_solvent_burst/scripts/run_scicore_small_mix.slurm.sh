#!/usr/bin/env bash
# Scicore smoke for the <=41-atom water/methanol mixture matrix.
# Submit from the repository root (or set MMML_REPO_ROOT):
#   MMML_CKPT=/path/to/checkpoint sbatch workflows/pbc_solvent_burst/scripts/run_scicore_small_mix.slurm.sh
# The first cell is the 2:2 mixture; the remaining 1:3 and 3:1 cells run only
# after it returns successfully.
#SBATCH --job-name=mmml-small-mix
#SBATCH --partition=scicore
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/diagnostics/scicore-small-mix-%j.out
#SBATCH --error=artifacts/diagnostics/scicore-small-mix-%j.err

set -euo pipefail
REPO_ROOT="${MMML_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$REPO_ROOT"
source scripts/resolve_mmml_env.sh
mmml_resolve_env "$REPO_ROOT"
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64
CFG="$REPO_ROOT/workflows/pbc_solvent_burst/config.scicore-small-mix.yaml"
RUNNER="$REPO_ROOT/workflows/pbc_solvent_burst/scripts/run_job.py"
[[ -n "${MMML_CKPT:-}" ]] || { echo 'MMML_CKPT is required' >&2; exit 2; }

# Run balanced 2:2 first as the representative smoke; only then test the two
# asymmetric compositions. Each run writes an independent output directory.
for tag in meohx50_tip3x50_4 meohx75_tip3x25_4 meohx25_tip3x75_4; do
  MMML_PBC_OUTPUT_ROOT="$REPO_ROOT/artifacts/pbc_solvent_burst_scicore_small_mix/$tag" \
    "$MMML_PYTHON" "$RUNNER" --config "$CFG" --tag "$tag"
done
