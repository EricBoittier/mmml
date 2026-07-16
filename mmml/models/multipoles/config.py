"""Shared configuration for training and loading molecular multipole models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Model-shape options persisted alongside multipole checkpoints."""

    features: int = 64
    max_degree: int = 3
    target_degree: int | None = None
    num_iterations: int = 3
    num_basis_functions: int = 16
    cutoff: float = 6.0
    max_atomic_number: int = 118
    compose_dipole_from_atomic: bool = False
    enforce_total_charge: bool = True
