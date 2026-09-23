#!/usr/bin/env bash
# Queue polar-only learning-curve jobs (same valid NPZ, smaller train).
# Does not download data. Run on login12; this agent cannot sbatch for you.
#
#   scripts/spice_alpha/submit_efield_lc.sh
#   SUBMIT=0 scripts/spice_alpha/submit_efield_lc.sh   # splits only
#
# Leaves 22868157 (full-train ENERGY_WEIGHT=0) alone. Unique CKPT per frac.

set -euo pipefail

ROOT="${MMML_REPO:-$HOME/mmml}"
cd "$ROOT"
SRC_SPLITS="${SRC_SPLITS:-$HOME/data/spicealpha/mmml_efield_full/splits_des_mono}"
OUT_ROOT="${OUT_ROOT:-$HOME/data/spicealpha/mmml_efield_lc}"
FRACTIONS="${FRACTIONS:-0.01,0.03,0.10,0.30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-0}"
MODE="${MODE:-big}"
EPOCHS="${EPOCHS:-100}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-0}"
POLAR_WEIGHT="${POLAR_WEIGHT:-100}"
RESTART="${RESTART:-$HOME/mmml/ckpts/spice_ef_polar_big/params-best-441a9161-9eca-4563-9957-04c9d2ec5a34.json}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/ckpts}"
SUBMIT="${SUBMIT:-1}"

if [[ ! -d "$SRC_SPLITS" ]]; then
  echo "SRC_SPLITS is not a directory: $SRC_SPLITS" >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/spice_alpha/make_efield_lc_splits.py" \
  "$SRC_SPLITS" "$OUT_ROOT" \
  --fracs "$FRACTIONS" --batch-size "$BATCH_SIZE" --seed "$SEED"

if [[ "$SUBMIT" == "0" ]]; then
  echo "SUBMIT=0: splits only under $OUT_ROOT"
  exit 0
fi

IFS=',' read -r -a FRAC_ARR <<< "$FRACTIONS"
for frac in "${FRAC_ARR[@]}"; do
  frac="$(echo "$frac" | tr -d ' ')"
  tag="$(python -c "from mmml.data.efield_lc import frac_tag; print(frac_tag(float('$frac')))")"
  splits="$OUT_ROOT/splits_$tag"
  ckpt="$CKPT_ROOT/spice_ef_polar_lc_$tag"
  echo "sbatch $tag frac=$frac splits=$splits ckpt=$ckpt"
  sbatch --partition=rtx4090 --qos=rtx4090-6hours --time=06:00:00 \
    --job-name="spice-ef-lc-$tag" \
    --export=ALL,MODE="$MODE",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",ENERGY_WEIGHT="$ENERGY_WEIGHT",POLAR_WEIGHT="$POLAR_WEIGHT",RESTART="$RESTART",SPLITS="$splits",CKPT="$ckpt" \
    "$ROOT/scripts/spice_alpha/train_efield_polar.sbatch"
done
