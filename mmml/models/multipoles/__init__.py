"""Equivariant molecular multipole models and representation utilities."""

from .model import E3xMultipoleModel
from .representations import (
    irrep_blocks_to_traceless,
    split_irrep_blocks,
    traceless_tensors_from_irreps,
)

__all__ = [
    "E3xMultipoleModel",
    "irrep_blocks_to_traceless",
    "split_irrep_blocks",
    "traceless_tensors_from_irreps",
]
