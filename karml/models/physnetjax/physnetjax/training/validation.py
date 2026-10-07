"""Validation helpers for PhysNetJax training inputs."""

from __future__ import annotations

import numpy as np


def validate_atomic_numbers(train_data, valid_data, model_max_atomic_number):
    """Fail fast when dataset atomic numbers are incompatible with model embeddings."""
    datasets = {
        "train_data": train_data,
        "valid_data": valid_data,
    }
    max_seen = -1

    for dataset_name, dataset in datasets.items():
        if "Z" not in dataset:
            raise KeyError(f"{dataset_name} is missing required key 'Z'.")

        z_values = np.asarray(dataset["Z"])
        if z_values.size == 0:
            raise ValueError(f"{dataset_name}['Z'] is empty.")

        z_min = int(np.min(z_values))
        z_max = int(np.max(z_values))
        max_seen = max(max_seen, z_max)

        if z_min < 0:
            raise ValueError(
                f"{dataset_name} contains negative atomic numbers (min={z_min}). "
                "Atomic numbers must be >= 0."
            )

    if max_seen > int(model_max_atomic_number):
        raise ValueError(
            "Dataset atomic numbers exceed model.max_atomic_number: "
            f"max(Z)={max_seen}, model.max_atomic_number={model_max_atomic_number}. "
            "Increase EF(max_atomic_number=...) to at least max(Z)."
        )
