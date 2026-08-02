"""ASE constraints for mode-check geometry control."""

from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixConstraint

from .bonds import monomer_slices


class FixMonomerCOMs(FixConstraint):
    """Keep each monomer's mass-weighted COM fixed at capture / target positions.

    Relaxes intramolecular degrees of freedom under FIRE without drifting the
    hybrid COM handoff distance (per-monomer analogue of ASE ``FixCom``).
    """

    def __init__(
        self,
        atoms: Atoms,
        atoms_per_monomer: list[int] | tuple[int, ...],
        *,
        target_coms: np.ndarray | None = None,
    ):
        self.atoms_per_monomer = [int(n) for n in atoms_per_monomer]
        if int(sum(self.atoms_per_monomer)) != len(atoms):
            raise ValueError(
                f"atoms_per_monomer sum ({sum(self.atoms_per_monomer)}) != "
                f"natoms ({len(atoms)})"
            )
        self.slices = monomer_slices(self.atoms_per_monomer)
        if target_coms is None:
            self.target_coms = np.stack(
                [_com_of_slice(atoms, sl) for sl in self.slices],
                axis=0,
            )
        else:
            self.target_coms = np.asarray(target_coms, dtype=float)
            if self.target_coms.shape != (len(self.slices), 3):
                raise ValueError(
                    f"target_coms shape {self.target_coms.shape} != "
                    f"({len(self.slices)}, 3)"
                )

    def get_removed_dof(self, atoms) -> int:  # noqa: ANN001
        return 3 * len(self.slices)

    def adjust_positions(self, atoms, new):  # noqa: ANN001
        masses = atoms.get_masses()
        for i, sl in enumerate(self.slices):
            m = masses[sl]
            new_cm = m @ new[sl] / m.sum()
            new[sl] += self.target_coms[i] - new_cm

    def adjust_forces(self, atoms, forces):  # noqa: ANN001
        # Same Lagrange form as ase.constraints.FixCom, per monomer.
        masses = atoms.get_masses()
        for sl in self.slices:
            m = masses[sl]
            lmd = m @ forces[sl] / float(np.sum(m**2))
            forces[sl] -= m[:, None] * lmd

    def adjust_momenta(self, atoms, momenta):  # noqa: ANN001
        masses = atoms.get_masses()
        for sl in self.slices:
            m = masses[sl]
            v_com = momenta[sl].sum(axis=0) / float(m.sum())
            momenta[sl] -= m[:, None] * v_com

    def todict(self) -> dict:
        return {
            "name": "FixMonomerCOMs",
            "kwargs": {
                "atoms_per_monomer": list(self.atoms_per_monomer),
                "target_coms": self.target_coms.tolist(),
            },
        }


def _com_of_slice(atoms: Atoms, sl: slice) -> np.ndarray:
    indices = list(range(*sl.indices(len(atoms))))
    return np.asarray(atoms.get_center_of_mass(indices=indices), dtype=float)
