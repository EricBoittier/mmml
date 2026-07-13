#!/usr/bin/env bash
# Submit with: sbatch scripts/run_spooky_smoke_slurm.sh
# One-cell, end-to-end ML/MM smoke test using config.spooky-smoke.yaml.
#SBATCH --job-name=mmml-spooky-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=00:30:00
#SBATCH --output=../../artifacts/pbc_solvent_burst_spooky_smoke/slurm-%j.out
#SBATCH --error=../../artifacts/pbc_solvent_burst_spooky_smoke/slurm-%j.err

set -euo pipefail

<<<<<<< HEAD
<<<<<<< HEAD
# Slurm executes a staged copy under /var/spool; retain the submit directory.
WORKFLOW_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
=======
WORKFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
>>>>>>> c59714611 (asdf)
=======
# Slurm executes a staged copy under /var/spool; retain the submit directory.
WORKFLOW_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
>>>>>>> cc7dd8f95 (sdaf)
REPO_ROOT="$(cd "$WORKFLOW_ROOT/../.." && pwd)"
CONFIG="${MMML_SMOKE_CONFIG:-$WORKFLOW_ROOT/config.spooky-smoke.yaml}"
cd "$REPO_ROOT"

# shellcheck source=../../../scripts/resolve_mmml_env.sh
source "$REPO_ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$REPO_ROOT"
export JAX_ENABLE_X64=1

# DCM:10 contains 100 ML atoms; the installed default PBC pair-buffer tier is sufficient.
eval "$("$REPO_ROOT/scripts/ensure_charmm_mlpot_limits.sh" --n-ml 100 --pbc --box-size 32 \
  | tee /dev/stderr | grep '^export ')"

mkdir -p "$REPO_ROOT/artifacts/pbc_solvent_burst_spooky_smoke"
exec "$MMML_PYTHON" "$WORKFLOW_ROOT/scripts/run_job.py" --tag dcm_10 --config "$CONFIG"
