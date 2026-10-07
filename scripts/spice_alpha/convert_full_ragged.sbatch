#!/bin/bash
#SBATCH --job-name=ragged-conv
#SBATCH --partition=scicore
#SBATCH --qos=6hours
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --array=0-11
# Convert each SPICE-α HDF5 file to a ragged NPZ shard (one array task per file).
set -euo pipefail
REPO="${KARML_REPO:-$HOME/karml_dev}"
PY="${KARML_PYTHON:-$HOME/karml_new/.venv/bin/python}"
H5_DIR="${H5_DIR:-$HOME/data/spicealpha/SPICE-alpha}"
OUT_DIR="${OUT_DIR:-$HOME/data/spicealpha/ragged}"
cd "$REPO"
mapfile -t STEMS < <(PYTHONPATH="$REPO" "$PY" -c "from karml.data.spice_alpha_ragged import SUBSETS; print('\n'.join(sorted(SUBSETS)))")
PYTHONPATH="$REPO" "$PY" scripts/spice_alpha/convert_full_ragged.py "$H5_DIR" "$OUT_DIR" --only "${STEMS[$SLURM_ARRAY_TASK_ID]}"
