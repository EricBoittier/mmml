#!/bin/bash
# Quick-start script for PhysNet training with default settings

python trainer.py \
    --train ../preclassified_data/energies_forces_dipoles_train.npz \
    --valid ../preclassified_data/energies_forces_dipoles_valid.npz \
    --name co2_physnet_default \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --no-tensorboard --energy-unit hartree --convert-energy-to eV --subtract-atomic-energies
