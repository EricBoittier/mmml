from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistanceRestraint:
    """Harmonic distance restraint; energy is ``0.5*k*(r-target)^2``."""

    indices: tuple[int, int]
    target_A: float
    k_ev_A2: float

    def __post_init__(self) -> None:
        if self.indices[0] == self.indices[1] or min(self.indices) < 0:
            raise ValueError("distance restraint requires two distinct non-negative atom indices")
        if self.target_A < 0 or self.k_ev_A2 < 0:
            raise ValueError("distance target and force constant must be non-negative")

    def energy(self, positions):
        import jax.numpy as jnp

        distance = jnp.linalg.norm(positions[self.indices[0]] - positions[self.indices[1]])
        return 0.5 * self.k_ev_A2 * (distance - self.target_A) ** 2
