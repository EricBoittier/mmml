"""Reusable restraint specifications, separate from sampling protocols."""

from mmml.md.restraints.distance import DistanceRestraint
from mmml.md.restraints.dihedral import DihedralRestraint

__all__ = ["DistanceRestraint", "DihedralRestraint"]
