from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mmml.models.multipoles.electrostatics import (
    LearnedMolecularMultipoleElectrostatics,
)


class FixedChargeMultipoles(LearnedMolecularMultipoleElectrostatics):
    def __init__(self):
        Calculator.__init__(self)
        self.max_ell = 0
        self.softening_bohr = 0.0
        self.force_step_angstrom = 1.0e-5

    def predict_fragment_multipoles(self, atoms):
        positions_bohr = atoms.get_positions() / 0.529177210903
        zeros2 = np.zeros((2, 3))
        return {
            "fragments": [np.array([0]), np.array([1])],
            "origins_bohr": positions_bohr,
            "origins_angstrom": atoms.get_positions(),
            "multipoles": np.zeros((2, 16)),
            "charges": np.array([1.0, -1.0]),
            "dipoles_bohr": zeros2,
            "quadrupoles_bohr": np.zeros((2, 3, 3)),
            "octupoles_bohr": np.zeros((2, 3, 3, 3)),
        }


def test_learned_multipole_calculator_returns_conservative_fd_forces():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms.calc = FixedChargeMultipoles()

    forces = atoms.get_forces()

    assert forces.shape == (2, 3)
    np.testing.assert_allclose(forces[0], -forces[1], atol=1.0e-8)
    assert forces[0, 0] > 0.0
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1.0e-8)
    assert atoms.calc.results["force_method"] == "central_finite_difference"


def test_learned_multipole_forces_with_atoms_none_restores_geometry():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    calc = FixedChargeMultipoles()
    calc.atoms = atoms.copy()
    original = atoms.get_positions().copy()

    calc.calculate(atoms=None, properties=("energy", "forces"))

    np.testing.assert_allclose(calc.atoms.get_positions(), original)
    assert "forces" in calc.results
    assert calc.results["forces"].shape == (2, 3)
