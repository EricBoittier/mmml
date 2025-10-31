#!/bin/bash
# Quick-start script for DCMNet training with default settings

python trainer.py \
    --train-efd ../preclassified_data/energies_forces_dipoles_train.npz \
    --train-grid ../preclassified_data/grids_esp_train.npz \
    --valid-efd ../preclassified_data/energies_forces_dipoles_valid.npz \
    --valid-grid ../preclassified_data/grids_esp_valid.npz \
    --name co2_dcmnet_default \
    --epochs 100 \
    --batch-size 32 \
    --n-dcm 3

