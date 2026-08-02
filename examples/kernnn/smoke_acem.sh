#!/usr/bin/env bash
# Smoke-sized ACEM KerNN + PhysNet-teacher run (few epochs).
set -euo pipefail
ROOT="${ROOT:-$HOME/abirh}"
ACEM_SPLITS="${ACEM_SPLITS:-$ROOT/splits/acem}"
PHYSNET_ACEM="${PHYSNET_ACEM:-$ROOT/ckpts/run/params_acem1_2026-07-29_04-54-20.json}"
OUT="${OUT:-$ROOT/artifacts/kernnn_smoke}"
mkdir -p "$OUT"

mmml physnet-evaluate \
  --checkpoint "$PHYSNET_ACEM" \
  --data "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  -o "$OUT/physnet_eval" \
  --batch-size 16 \
  --num-samples 64 \
  --plots

mmml kernnn-train \
  --distance-scheme acem \
  --train-npz "$ACEM_SPLITS/energies_forces_dipoles_train.npz" \
  --valid-npz "$ACEM_SPLITS/energies_forces_dipoles_valid.npz" \
  --teacher-checkpoint "$PHYSNET_ACEM" \
  --distill-alpha 0.5 \
  --n-hidden 32 \
  --batch-size 16 \
  --epochs 5 \
  --patience 5 \
  --workdir "$OUT/acem_distill_smoke"

mmml kernnn-evaluate \
  --checkpoint "$OUT/acem_distill_smoke/best.json" \
  --data "$ACEM_SPLITS/energies_forces_dipoles_test.npz" \
  --split all \
  --batch-size 16 \
  --output-dir "$OUT/acem_distill_smoke/eval"
