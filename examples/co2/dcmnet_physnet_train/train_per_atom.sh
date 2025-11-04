#!/bin/bash
#
# Example: Train PhysNetJax with per-atom energy scaling
#
# This script demonstrates training with energies scaled by number of atoms.
# This normalizes energies to be comparable across molecules of different sizes.
#

python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --name co2_physnet_per_atom \
  --batch-size 32 \
  --epochs 100 \
  --learning-rate 0.001 \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --scale-by-atoms \
  --verbose

