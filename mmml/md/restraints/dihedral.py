from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DihedralRestraint:
    """Harmonic periodic dihedral restraint (force constant in eV/rad²)."""

    indices: tuple[int, int, int, int]
    target_deg: float
    k_ev: float

    def __post_init__(self) -> None:
        if len(set(self.indices)) != 4 or min(self.indices) < 0:
            raise ValueError("dihedral restraint requires four distinct non-negative atom indices")
        if self.k_ev < 0:
            raise ValueError("dihedral force constant must be non-negative")
