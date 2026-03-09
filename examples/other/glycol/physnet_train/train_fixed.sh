#!/bin/bash
#
# FIXED: Train PhysNetJax with proper energy preprocessing
#
# The key issue: absolute quantum energies (~-5104 eV) are too large!
# Solution: Subtract atomic energy references to get interaction energies
#

python trainer.py \
  --train ../preclassified_data/energies_forces_dipoles_train.npz \
  --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
  --name co2_physnet_fixed \
  --batch-size 32 \
  --epochs 100 \
  --learning-rate 0.001 \
  --energy-weight 1.0 \
  --forces-weight 50.0 \
  --dipole-weight 25.0 \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression \
  --no-tensorboard \
  --verbose

