"""Unit tests for ORCA QM reference backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mmml.interfaces.qc_backends.orca_qm import (
    OrcaQMBackend,
    parse_orca_out_energy,
    read_orca_engrad,
    render_orca_input,
)


def test_read_orca_engrad_gradient_shape(tmp_path: Path):
    natoms = 2
    grad = np.arange(3 * natoms, dtype=np.float64)
    lines = [str(natoms), "-10.0", *grad.tolist()]
    path = tmp_path / "test.engrad"
    path.write_text("\n".join(str(x) for x in lines))
    energy, g = read_orca_engrad(path)
    assert energy == pytest.approx(-10.0)
    assert g is not None
    assert g.shape == (natoms, 3)
    np.testing.assert_allclose(g.ravel(), grad)


def test_parse_orca_out_energy_total_energy_line():
    text = "blah\nTotal Energy       :    -123.4567890123 Eh\n"
    assert parse_orca_out_energy(text) == pytest.approx(-123.4567890123)


def test_orca_template_placeholders():
    atoms = Atoms("He", positions=[[0, 0, 0]])
    rendered = render_orca_input(
        atoms=atoms,
        charge=0,
        multiplicity=1,
        template="charge={charge} mult={mult}\n{xyz}\n",
        method="HF",
        basis="def2-TZVP",
        pal=4,
    )
    assert "charge=0" in rendered
    assert "mult=1" in rendered
    assert "He" in rendered


def test_orca_qm_forces_from_gradient():
    captured: list[Path] = []

    def fake_run(atoms, workdir, backend):
        captured.append(workdir)
        (workdir / "job.engrad").write_text(
            "3\n-1.0\n" + "\n".join("0.01" for _ in range(9)) + "\n"
        )
        return read_orca_engrad(workdir / "job.engrad")

    backend = OrcaQMBackend(run_orca=fake_run)
    out = backend.evaluate_batch([Atoms("H2O")], properties=frozenset({"energy", "forces"}))
    assert out["F"].shape == (1, 3, 3)
    assert out["F"][0, 0, 0] == pytest.approx(-0.01)
