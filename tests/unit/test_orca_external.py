"""Unit tests for the MMML ORCA external-tool wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mmml.interfaces.orca_external.protocol import read_extinp, write_engrad
from mmml.interfaces.orca_external.runner import (
    MmmlOrcaExternalRunner,
    atoms_from_xyz,
    clear_calculator_cache,
    evaluate_structure,
    mmml_forces_to_orca_gradient,
)
from mmml.interfaces.orca_external.settings import MmmlOrcaSettings


def test_read_extinp_parses_optional_pointcharges(tmp_path: Path) -> None:
    xyz_path = tmp_path / "job_EXT.xyz"
    xyz_path.write_text("1\ncomment\nH 0.0 0.0 0.0\n")
    pc_path = tmp_path / "pointcharges.pc"
    pc_path.write_text("")

    extinp_path = tmp_path / "job_EXT.extinp.tmp"
    extinp_path.write_text(
        "\n".join(
            [
                "job_EXT.xyz",
                "0",
                "1",
                "4",
                "1",
                "pointcharges.pc",
            ]
        )
    )

    extinp = read_extinp(extinp_path)
    assert extinp.xyz_path == xyz_path.resolve()
    assert extinp.charge == 0
    assert extinp.multiplicity == 1
    assert extinp.ncores == 4
    assert extinp.do_gradient is True
    assert extinp.pointcharges_path == pc_path.resolve()


def test_write_engrad_matches_orca_layout(tmp_path: Path) -> None:
    out_path = tmp_path / "job_EXT.engrad"
    write_engrad(
        out_path,
        natoms=2,
        energy_hartree=-1.25,
        gradient_hartree_bohr=[0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
    )

    text = out_path.read_text()
    assert "2\n" in text
    assert "-1.250000000000e+00" in text
    assert " 1.000000000000e-01" in text
    assert " 3.000000000000e-01" in text


def test_mmml_forces_to_orca_gradient_sign_and_units() -> None:
    forces = np.array([[1.0, 0.0, 0.0]])
    gradient = mmml_forces_to_orca_gradient(forces)
    assert gradient.shape == (3,)
    assert gradient[0] < 0.0


def test_evaluate_structure_uses_mock_calculator() -> None:
    class _MockCalc:
        def calculate(self, atoms=None, properties=None, system_changes=None):
            self.results = {
                "energy": -13.6,
                "forces": np.array([[1.0, 0.0, 0.0]]),
            }

        def get_potential_energy(self, atoms=None):
            return self.results["energy"]

        def get_forces(self, atoms=None):
            return self.results["forces"]

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    energy, gradient = evaluate_structure(atoms, _MockCalc(), do_gradient=True)

    assert energy == pytest.approx(-13.6 / 27.211386)
    assert len(gradient) == 3
    assert gradient[0] < 0.0


def test_runner_writes_engrad_with_mocked_checkpoint(tmp_path: Path, monkeypatch) -> None:
    clear_calculator_cache()

    xyz_path = tmp_path / "water_EXT.xyz"
    xyz_path.write_text(
        "\n".join(
            [
                "2",
                "water",
                "O 0.0 0.0 0.0",
                "H 0.96 0.0 0.0",
            ]
        )
    )
    extinp_path = tmp_path / "water_EXT.extinp.tmp"
    extinp_path.write_text(
        "\n".join(
            [
                "water_EXT.xyz",
                "0",
                "1",
                "1",
                "1",
            ]
        )
    )

    class _MockCalc:
        def calculate(self, atoms=None, properties=None, system_changes=None):
            self.results = {
                "energy": -100.0,
                "forces": np.zeros((len(atoms), 3)),
            }

        def get_potential_energy(self, atoms=None):
            return self.results["energy"]

        def get_forces(self, atoms=None):
            return self.results["forces"]

    def _fake_get_calculator(settings: MmmlOrcaSettings):
        return _MockCalc()

    monkeypatch.setattr(
        "mmml.interfaces.orca_external.runner.get_calculator",
        _fake_get_calculator,
    )

    settings = MmmlOrcaSettings(checkpoint=tmp_path / "dummy.pkl")
    engrad_path = MmmlOrcaExternalRunner(settings).run(extinp_path)

    assert engrad_path == tmp_path / "water_EXT.engrad"
    assert engrad_path.is_file()
    assert "2\n" in engrad_path.read_text()

    clear_calculator_cache()


def test_atoms_from_xyz_round_trip(tmp_path: Path) -> None:
    xyz_path = tmp_path / "h2o.xyz"
    xyz_path.write_text("3\n\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")
    atoms = atoms_from_xyz(xyz_path)
    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
