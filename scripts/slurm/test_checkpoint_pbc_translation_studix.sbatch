#!/usr/bin/env bash
#SBATCH --job-name=pbc-repeat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/npt_argon_water/checkpoint_pes_compare_20260802/spooky_charges_epoch3/pbc-repeat-%j.out
#SBATCH --error=artifacts/npt_argon_water/checkpoint_pes_compare_20260802/spooky_charges_epoch3/pbc-repeat-%j.err
set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
source "$ROOT/scripts/resolve_mmml_env.sh"
mmml_resolve_env "$ROOT"
export JAX_ENABLE_X64=1
export MMML_ML_DTYPE=float64
export XLA_PYTHON_CLIENT_PREALLOCATE=false

BOX="$ROOT/artifacts/lj_scales_des_validation/boxes/tip3"
OUT="$ROOT/artifacts/npt_argon_water/checkpoint_pes_compare_20260802/spooky_charges_epoch3"
CHECKPOINT_REL="${CHECKPOINT_REL:-artifacts/spooky_so3lr_charges/epoch-0003_params.json}"
OUTPUT_NAME="${OUTPUT_NAME:-pbc_translation_repeat}"
MM_CHARGE_MODE="${MM_CHARGE_MODE:-fixed}"
EXTRA_ARGS=()
if [[ "${REPEAT_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--repeat-only)
fi
"$MMML_PYTHON" scripts/test_checkpoint_pbc_translation.py \
  --pdb "$BOX/model.pdb" \
  --psf "$BOX/model.psf" \
  --checkpoint "$ROOT/$CHECKPOINT_REL" \
  --output "$OUT/$OUTPUT_NAME.json" \
  --box 28.0 --n-monomers 732 --ml-batch-size 64 \
  --mm-charge-mode "$MM_CHARGE_MODE" \
  "${EXTRA_ARGS[@]}"
