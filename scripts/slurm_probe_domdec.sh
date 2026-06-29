#!/usr/bin/env bash
#SBATCH --job-name=domdec_probe
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=domdec_probe_%j.log
#SBATCH --error=domdec_probe_%j.err
#
# Run the DOMDEC atom-map probe on a compute node (proper MPI / InfiniBand).
#
# Usage:
#   cd ~/tests/boxes/domdec_dcm80_l82
#   sbatch ~/mmml/scripts/slurm_probe_domdec.sh \
#       --psf $PWD/model.psf --crd $PWD/model.crd --box 82 --ndir 8 --cutnb 10
#
# Or for the dense preset:
#   cd ~/tests/boxes/domdec_dcm200_l50
#   sbatch ~/mmml/scripts/slurm_probe_domdec.sh \
#       --psf $PWD/model.psf --crd $PWD/model.crd --box 50 --ndir 8 --cutnb 6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate mmml venv (adjust path if needed)
source "${HOME}/mmml/.venv/bin/activate" 2>/dev/null || true

echo "Host      : $(hostname)"
echo "Date      : $(date)"
echo "SLURM job : ${SLURM_JOB_ID:-local}"
echo "Args      : $*"
echo ""

# Pass all remaining args straight to the probe script
mpirun -np "${SLURM_NTASKS:-8}" \
    python "${SCRIPT_DIR}/probe_domdec_atoms_live.py" "$@"

echo ""
echo "=== per-rank natoml ==="
grep "natoml=" domdec_probe_rank*.txt 2>/dev/null || echo "(no rank files found — check working directory)"

echo ""
echo "=== cross-rank disjoint (rank 0) ==="
grep -A3 "Cross-rank" domdec_probe_rank00.txt 2>/dev/null || true

echo ""
echo "=== rank 00 full ==="
cat domdec_probe_rank00.txt 2>/dev/null || true
