#!/usr/bin/env bash
#SBATCH --job-name=mmml-hetero-boundary
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000
#SBATCH --time=00:30:00
#SBATCH --output=../../artifacts/diagnostics/hetero-boundary-%j.out
#SBATCH --error=../../artifacts/diagnostics/hetero-boundary-%j.err
set -euo pipefail
ROOT="$(cd "${SLURM_SUBMIT_DIR:?}/../.." && pwd)"
cd "$ROOT"
export JAX_ENABLE_X64=1 MMML_ML_DTYPE=float64
exec "$ROOT/.venv/bin/python" workflows/pbc_solvent_burst/scripts/scan_heterogeneous_dimer_boundary.py \
  --checkpoint artifacts/spooky_so3lr_muon_tight/epoch-0001_params.json \
  --dcm-pdb artifacts/pbc_heat_boundary_probe/dcm_10/pycharmm_init/02_mini.pdb \
  --aco-pdb mmml/generate/sample/pdb/aco_monomer.pdb \
  --output-dir artifacts/diagnostics/heterogeneous_boundary_muon_epoch1
