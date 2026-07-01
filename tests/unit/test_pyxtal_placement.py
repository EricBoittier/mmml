"""Unit tests for PyXtal placement helpers (mocked; no pyxtal required)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def test_have_pyxtal_false_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyxtal" or name.startswith("pyxtal."):
            raise ImportError("no pyxtal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from mmml.interfaces.pyxtal_placement import have_pyxtal

    assert have_pyxtal() is False


def test_parse_supercell_reps():
    from mmml.interfaces.pyxtal_placement import parse_supercell_reps

    assert parse_supercell_reps("2,2,2") == (2, 2, 2)
    assert parse_supercell_reps("2x1x3") == (2, 1, 3)


def test_parse_stoichiometry_defaults_and_repeat():
    from mmml.interfaces.pyxtal_placement import parse_stoichiometry

    assert parse_stoichiometry(["a.xyz", "b.xyz"], None, [3]) == [3, 3]
    assert parse_stoichiometry(["a.xyz"], None, None) == [2]


def test_crystal_mass_density_and_scale():
    from ase import Atoms

    from mmml.interfaces.pyxtal_placement import (
        crystal_mass_density_g_cm3,
        scale_atoms_cell_to_density,
    )

    atoms = Atoms("C4", positions=[[0, 0, 0]] * 4, cell=[5, 5, 5], pbc=True)
    rho0 = crystal_mass_density_g_cm3(atoms)
    target = rho0 / 2.0
    scale_atoms_cell_to_density(atoms, target)
    rho1 = crystal_mass_density_g_cm3(atoms)
    assert rho1 == pytest.approx(target, rel=1e-9)
    assert atoms.get_volume() == pytest.approx((5.0 * (rho0 / target) ** (1 / 3)) ** 3, rel=1e-6)


def test_scale_atoms_cell_rejects_bad_density():
    from ase import Atoms

    from mmml.interfaces.pyxtal_placement import scale_atoms_cell_to_density

    atoms = Atoms("C", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    with pytest.raises(ValueError, match="positive"):
        scale_atoms_cell_to_density(atoms, 0.0)


def test_build_molecular_crystal_random_mock(tmp_path, monkeypatch):
    from ase import Atoms

    from mmml.interfaces.pyxtal_placement import (
        MolecularCrystalBuildRequest,
        build_molecular_crystal_random,
    )

    class FakePyxtal:
        valid = True
        formula = "C2H2"
        group = mock.Mock(number=14)

        def __init__(self, molecular=True):
            self.molecular = molecular

        def from_random(self, dim, spg, mols, zs, factor, seed):
            assert dim == 3 and spg == 14 and mols == ["x.xyz"] and zs == [2]

        def to_ase(self, resort=True, center_only=False):
            return Atoms(
                symbols="CC",
                positions=[[0, 0, 0], [1.3, 0, 0]],
                cell=np.diag([5.0, 5.0, 5.0]),
                pbc=True,
            )

    monkeypatch.setattr(
        "mmml.interfaces.pyxtal_placement._import_pyxtal",
        lambda: FakePyxtal,
    )
    req = MolecularCrystalBuildRequest(
        molecules=["x.xyz"],
        stoichiometry=[2],
        space_group=14,
        seed=1,
        max_attempts=3,
    )
    result = build_molecular_crystal_random(req)
    assert result.attempts == 1
    assert len(result.atoms) == 2
    assert result.atoms.pbc.all()


def test_atoms_to_reference_npz(tmp_path):
    from ase import Atoms

    from mmml.interfaces.pyxtal_placement import atoms_to_reference_npz

    atoms = Atoms(
        symbols="HOH",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    out = atoms_to_reference_npz(atoms, tmp_path / "ref.npz")
    data = np.load(out)
    assert data["R"].shape == (3, 3)
    assert data["Z"].tolist() == [1, 8, 1]
    assert "cell" in data


def test_optimize_ase_atoms_emt():
    from ase import Atoms

    from mmml.interfaces.aseInterface.pyxtal_optimize import optimize_ase_atoms

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)

    class FakeCalc:
        def get_forces(self, atoms=None):
            return np.zeros((2, 3))

        def get_potential_energy(self):
            return 0.0

    class FakeOpt:
        nsteps = 3

        def __init__(self, atoms, logfile=None):
            self.atoms = atoms

        def run(self, fmax=0.05, steps=200):
            return None

    def attach(a):
        a.calc = FakeCalc()
        return a

    with mock.patch(
        "mmml.interfaces.aseInterface.pyxtal_optimize.attach_emt_calculator",
        side_effect=attach,
    ), mock.patch(
        "mmml.interfaces.aseInterface.pyxtal_optimize._optimizer_class",
        return_value=FakeOpt,
    ):
        result = optimize_ase_atoms(atoms, use_emt=True, max_steps=5)
    assert result.optimizer == "bfgs"
    assert result.n_steps == 3
