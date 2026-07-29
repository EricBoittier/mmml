#!/usr/bin/env bash
# Example workflows: PhysNet evaluate + KerNN train (ACEM / FORM)
# Run from the directory that contains the NPZs / splits (e.g. ~/abirh).
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit these for your machine
# ---------------------------------------------------------------------------
ROOT="${ROOT:-$HOME/abirh}"
ACEM_FULL="${ACEM_FULL:-$ROOT/acem_mp2_aug-cc-pvtz_16106.npz}"
FORM_FULL="${FORM_FULL:-$ROOT/form_mp2_aug-cc-pvtz_4000.npz}"
ACEM_SPLITS="${ACEM_SPLITS:-$ROOT/splits/acem}"
# Portable PhysNet JSON you already trained:
PHYSNET_ACEM="${PHYSNET_ACEM:-$ROOT/ckpts/run/params_acem1_2026-07-29_04-54-20.json}"
# Or an Orbax training root / epoch dir:
# PHYSNET_ACEM="$ROOT/ckpts/run/acem1-23762349-..."

OUT="${OUT:-$ROOT/artifacts/kernnn}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# 1) Evaluate the existing PhysNet teacher on ACEM test split
# ---------------------------------------------------------------------------
echo "=== PhysNet evaluate (ACEM test) ==="
mmml physnet-evaluate \
  --checkpoint "$PHYSNET_ACEM" \
  --data "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  -o "$OUT/physnet_acem_eval" \
  --batch-size 32 \
  --plots

# ---------------------------------------------------------------------------
# 2) Train KerNN on ACEM (9 atoms, all-pairs distances) from existing splits
# ---------------------------------------------------------------------------
echo "=== KerNN train ACEM (ground truth only) ==="
mmml kernnn-train \
  --distance-scheme acem \
  --architecture ffnet \
  --n-hidden 64 \
  --batch-size 64 \
  --learning-rate 0.003 \
  --f-weight 10 \
  --epochs 500 \
  --patience 80 \
  --seed 42 \
  --train-npz "$ACEM_SPLITS/energies_forces_dipoles_train.npz" \
  --valid-npz "$ACEM_SPLITS/energies_forces_dipoles_valid.npz" \
  --test-npz  "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  --workdir "$OUT/acem_gt"

echo "=== KerNN evaluate ACEM test ==="
mmml kernnn-evaluate \
  --checkpoint "$OUT/acem_gt/best.json" \
  --data "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  --split all \
  --output-dir "$OUT/acem_gt/eval_test"

# ---------------------------------------------------------------------------
# 3) Train KerNN with PhysNet as teacher (distillation)
#    loss = alpha * MSE(GT) + (1-alpha) * MSE(teacher)
#    alpha=0.5 → equal GT / teacher; alpha=0 → pure teacher
# ---------------------------------------------------------------------------
echo "=== KerNN train ACEM (PhysNet teacher, distill_alpha=0.5) ==="
mmml kernnn-train \
  --distance-scheme acem \
  --architecture ffnet \
  --n-hidden 64 \
  --batch-size 32 \
  --learning-rate 0.003 \
  --f-weight 10 \
  --epochs 500 \
  --patience 80 \
  --seed 42 \
  --train-npz "$ACEM_SPLITS/energies_forces_dipoles_train.npz" \
  --valid-npz "$ACEM_SPLITS/energies_forces_dipoles_valid.npz" \
  --test-npz  "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  --teacher-checkpoint "$PHYSNET_ACEM" \
  --distill-alpha 0.5 \
  --workdir "$OUT/acem_distill"

mmml kernnn-evaluate \
  --checkpoint "$OUT/acem_distill/best.json" \
  --data "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  --split all \
  --output-dir "$OUT/acem_distill/eval_test"

# ---------------------------------------------------------------------------
# 4) FORM (6 atoms) from a single full NPZ (KerNN does the split)
#    If you have form splits already, prefer --train-npz/--valid-npz.
# ---------------------------------------------------------------------------
if [[ -f "$FORM_FULL" ]]; then
  echo "=== KerNN train FORM (single NPZ split) ==="
  mmml kernnn-train \
    --distance-scheme form \
    --data "$FORM_FULL" \
    --ntrain 3200 \
    --nvalid 400 \
    --seed 42 \
    --n-hidden 48 \
    --batch-size 64 \
    --epochs 400 \
    --patience 60 \
    --workdir "$OUT/form_gt"

  mmml kernnn-evaluate \
    --checkpoint "$OUT/form_gt/best.json" \
    --data "$FORM_FULL" \
    --split-json "$OUT/form_gt/data_split.json" \
    --split test \
    --output-dir "$OUT/form_gt/eval_test"
fi

echo "Done. Artifacts under $OUT"
