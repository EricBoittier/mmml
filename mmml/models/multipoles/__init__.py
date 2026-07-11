"""Equivariant molecular multipole models and representation utilities."""

from .model import (
    E3xDegreeMultipoleModel,
    E3xDipoleModel,
    E3xMultipoleModel,
    E3xOctupoleModel,
    E3xQuadrupoleModel,
)
from .representations import (
    irrep_blocks_to_traceless,
    split_irrep_blocks,
    traceless_tensors_from_irreps,
)

__all__ = [
    "E3xDegreeMultipoleModel",
    "E3xDipoleModel",
    "E3xMultipoleModel",
    "E3xOctupoleModel",
    "E3xQuadrupoleModel",
    "irrep_blocks_to_traceless",
    "split_irrep_blocks",
    "traceless_tensors_from_irreps",
]
