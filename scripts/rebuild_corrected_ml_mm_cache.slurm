#!/usr/bin/env bash
#SBATCH --job-name=mlmm-cache-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/diagnostics/mlmm-cache-v2-%j.out
#SBATCH --error=artifacts/diagnostics/mlmm-cache-v2-%j.err
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""

SOURCE="$ROOT/data/orbax_cache_so3lr/so3lr_train_flat_2ef9214c00a78127"
ENRICHED="$ROOT/data/orbax_cache_des_ml_mm_v2"
SPLITS="$ROOT/data/splits_des_ml_mm_v2"

"$MMML_PYTHON" scripts/prepare_ml_mm_dataset.py \
  --cache-dir "$SOURCE" --output-cache "$ENRICHED" \
  --num-workers "${SLURM_CPUS_PER_TASK}"

"$MMML_PYTHON" scripts/split_and_inspect_ml_mm_dataset.py \
  --cache-dir "$ENRICHED" --output-dir "$SPLITS"

"$MMML_PYTHON" scripts/audit_ml_mm_cache.py \
  "$SPLITS/train_cache" \
  --output "$ROOT/artifacts/diagnostics/des_train_cache_v2_audit.json"
