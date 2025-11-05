#!/bin/bash
#
# Example: Train PhysNetJax with atomic energy reference subtraction
#
# This script demonstrates training with atomic energy contributions removed.
# This is useful when you want the model to learn binding/interaction energies
# rather than total molecular energies.
#

python trainer.py \
  --train energies_forces_dipoles_train.npz \
  --valid energies_forces_dipoles_valid.npz \
  --name co2_physnet_atomic_refs \
  --batch-size 8 \
  --epochs 100 \
  --learning-rate 0.005 \
  --energy-weight 1.0 \
  --forces-weight 1.0 \
  --dipole-weight 25.0 \
   --energy-unit eV \
  --convert-energy-to eV \
  --schedule constant \
  --subtract-atomic-energies \
  --atomic-energy-method linear_regression \
  --natoms 3 \
  --verbose

