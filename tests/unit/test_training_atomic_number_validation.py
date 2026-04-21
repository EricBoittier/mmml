"""Unit tests for train-time atomic number validation."""

from __future__ import annotations

import numpy as np
import pytest

training = pytest.importorskip("mmml.models.physnetjax.physnetjax.training.training")


def _dataset(z_values: np.ndarray) -> dict[str, np.ndarray]:
    return {"Z": z_values.astype(np.int32)}


def test_validate_atomic_numbers_raises_when_train_z_exceeds_model_limit() -> None:
    with pytest.raises(ValueError, match="max\\(Z\\)=54"):
        training._validate_atomic_numbers(
            train_data=_dataset(np.array([[1, 6, 54]])),
            valid_data=_dataset(np.array([[1, 6, 8]])),
            model_max_atomic_number=36,
        )


def test_validate_atomic_numbers_raises_when_valid_z_exceeds_model_limit() -> None:
    with pytest.raises(ValueError, match="model.max_atomic_number=20"):
        training._validate_atomic_numbers(
            train_data=_dataset(np.array([[1, 6, 8]])),
            valid_data=_dataset(np.array([[1, 6, 21]])),
            model_max_atomic_number=20,
        )


def test_validate_atomic_numbers_raises_on_negative_z() -> None:
    with pytest.raises(ValueError, match="negative atomic numbers"):
        training._validate_atomic_numbers(
            train_data=_dataset(np.array([[1, -1, 8]])),
            valid_data=_dataset(np.array([[1, 6, 8]])),
            model_max_atomic_number=36,
        )


def test_validate_atomic_numbers_allows_supported_range() -> None:
    training._validate_atomic_numbers(
        train_data=_dataset(np.array([[0, 1, 8, 36]])),
        valid_data=_dataset(np.array([[1, 6, 20]])),
        model_max_atomic_number=36,
    )
