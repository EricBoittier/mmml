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
from .electrostatics import (
    AU_FIELD_TO_V_PER_ANGSTROM,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    LearnedMolecularMultipoleElectrostatics,
    field_on_slice,
    fragment_indices_from_atoms,
    pair_energy_charge_dipole_au,
    plot_field_summary,
)

__all__ = [
    "AU_FIELD_TO_V_PER_ANGSTROM",
    "BOHR_TO_ANGSTROM",
    "E3xDegreeMultipoleModel",
    "E3xDipoleModel",
    "E3xMultipoleModel",
    "E3xOctupoleModel",
    "E3xQuadrupoleModel",
    "HARTREE_TO_EV",
    "LearnedMolecularMultipoleElectrostatics",
    "field_on_slice",
    "fragment_indices_from_atoms",
    "irrep_blocks_to_traceless",
    "pair_energy_charge_dipole_au",
    "plot_field_summary",
    "split_irrep_blocks",
    "traceless_tensors_from_irreps",
]
