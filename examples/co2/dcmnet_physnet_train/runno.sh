#!/bin/bash

seed=${seed:-1}
echo "${seed}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

uv run python compare_models.py \
    --train-efd energies_forces_dipoles_train.npz \
    --train-esp grids_esp_train.npz \
    --valid-efd energies_forces_dipoles_valid.npz \
    --valid-esp grids_esp_valid.npz \
    --epochs 1000 \
    --batch-size 100 \
    --comparison-name "test${seed}" \
<<<<<<< HEAD
    -seed "${seed}"
=======
    --seed "${seed}"
>>>>>>> f5899bcf5dd3cccc2bb1f96f6f269611600b038b
