from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mmml.analysis.dimer_scans import (
    assign_mol_id,
    build_rigid_dimer,
    distance_scan_geometries,
    evaluate_scan,
    geometric_centroid,
    make_xtb_calculator,
    molecule_pair_labels,
)


class ConstantEnergyCalculator(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, energy_ev: float):
        super().__init__()
        self.energy_ev = energy_ev

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.energy_ev


def test_molecule_pair_labels_all_pairs_for_five_molecules():
    labels = ["DCM", "ACE", "BENZ", "TIP3", "MEOH"]

    pairs = molecule_pair_labels(labels)

    assert len(pairs) == 15
    assert pairs[0] == ("DCM", "DCM")
    assert pairs[-1] == ("MEOH", "MEOH")
    assert ("ACE", "TIP3") in pairs


def test_assign_mol_id_validates_fragment_sizes():
    atoms = Atoms("HeNeAr", positions=np.zeros((3, 3)))

    tagged = assign_mol_id(atoms, [1, 2])

    np.testing.assert_array_equal(tagged.arrays["mol_id"], np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="fragment sizes sum"):
        assign_mol_id(atoms, [1, 1])


def test_build_rigid_dimer_places_centroids_and_fragments():
    monomer_a = Atoms("H2", positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
    monomer_b = Atoms("He", positions=[[2.0, 1.0, 0.0]])

    dimer, fragments = build_rigid_dimer(
        monomer_a,
        monomer_b,
        distance_angstrom=5.0,
        axis=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(geometric_centroid(dimer[fragments[0]]), [0.0, 0.0, -2.5])
    np.testing.assert_allclose(geometric_centroid(dimer[fragments[1]]), [0.0, 0.0, 2.5])
    np.testing.assert_array_equal(dimer.arrays["mol_id"], np.array([0, 0, 1]))


def test_distance_scan_geometries_yields_metadata():
    monomer_a = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    monomer_b = Atoms("Cl", positions=[[0.0, 0.0, 0.0]])

    geometries = list(
        distance_scan_geometries(
            monomer_a,
            monomer_b,
            [3.0, 4.0],
            pair=("H", "Cl"),
        )
    )

    assert [geometry.distance_angstrom for geometry in geometries] == [3.0, 4.0]
    assert geometries[0].pair == ("H", "Cl")
    np.testing.assert_array_equal(geometries[0].fragments[0], np.array([0]))
    np.testing.assert_array_equal(geometries[0].fragments[1], np.array([1]))


def test_evaluate_scan_uses_calculator_factory():
    geometries = distance_scan_geometries(
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        [2.0],
        pair=("H", "H"),
    )

    rows = evaluate_scan(geometries, lambda: ConstantEnergyCalculator(energy_ev=0.5))

    assert rows == [
        {
            "molecule_a": "H",
            "molecule_b": "H",
            "distance_angstrom": 2.0,
            "energy_ev": 0.5,
            "energy_kcal_mol": pytest.approx(11.5302744335),
        }
    ]


def test_make_xtb_calculator_when_optional_dependency_is_available():
    pytest.importorskip("xtb", reason="xTB optional dependency is not installed")
    calculator = make_xtb_calculator()
    assert calculator is not None
