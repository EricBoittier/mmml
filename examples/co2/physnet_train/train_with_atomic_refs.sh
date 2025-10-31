#!/bin/bash
#
# Example: Train PhysNetJax with atomic energy reference subtraction
#
# This script demonstrates training with atomic energy contributions removed.
# This is useful when you want the model to learn binding/interaction energies
# rather than total molecular energies.
#

python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --name co2_physnet_atomic_refs \
  --batch-size 32 \
  --epochs 100 \
  --learning-rate 0.001 \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression \
  --verbose

