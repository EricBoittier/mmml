"""Tests for Packmol placement helpers (no Packmol binary required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_resolve_packmol_use_defaults():
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_sphere_use,
        resolve_packmol_use,
    )

    assert resolve_packmol_use(composition="MEOH:5") is False
    assert resolve_packmol_use(composition="MEOH:5", packmol=True) is True
    assert resolve_packmol_use(composition="MEOH:5", packmol=False) is False
    assert resolve_packmol_use(composition=None) is False
    assert resolve_packmol_sphere_use(composition="MEOH:5", packmol_radius=12.0) is False
    assert resolve_packmol_sphere_use(
        composition="MEOH:5", packmol=True, packmol_sphere=True
    ) is True
    assert resolve_packmol_sphere_use(composition=None, packmol_sphere=True) is False


def test_resolve_packmol_cube_side():
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_cube_side

    assert resolve_packmol_cube_side(box_size=38.0) == 38.0
    assert resolve_packmol_cube_side(packmol_radius=20.0) == 40.0
    assert resolve_packmol_cube_side(flat_bottom_radius=12.0) == 24.0
    with pytest.raises(ValueError, match="box-size"):
        resolve_packmol_cube_side()


def test_packmol_cube_origin():
    from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_cube_origin

    assert packmol_cube_origin((0.0, 0.0, 0.0), 40.0) == (-20.0, -20.0, -20.0)
    assert packmol_cube_origin((10.0, 0.0, -5.0), 20.0) == (0.0, -10.0, -15.0)


def test_resolve_packmol_sphere_radius_separate():
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_sphere_radius

    assert resolve_packmol_sphere_radius(25.0, 20.0) == 25.0
    assert resolve_packmol_sphere_radius(None, 20.0) == 20.0
    with pytest.raises(ValueError, match="packmol-radius"):
        resolve_packmol_sphere_radius(None, None)


def test_write_monomer_pdb_uses_psf_atomic_numbers(tmp_path):
    """Chlorinated residues (Z=17) must become ASE symbol Cl, not CL."""
    from ase.io import read as ase_read

    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        write_monomer_pdb_for_packmol,
    )

    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    z = np.array([17, 6], dtype=int)
    pdb_path = tmp_path / "dcm.pdb"
    write_monomer_pdb_for_packmol(pdb_path, coords, z)
    symbols = ase_read(pdb_path).get_chemical_symbols()
    assert symbols == ["Cl", "C"]


def test_write_monomer_pdb_charmm_names_not_carbon_for_cl(tmp_path):
    """Regression: coord loop variable must not shadow atomic_numbers (CL1 -> Cl)."""
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        write_monomer_pdb_for_packmol,
    )

    coords = np.zeros((5, 3), dtype=float)
    Z = np.array([6, 1, 1, 17, 17], dtype=int)
    names = ["C", "H1", "H2", "CL1", "CL2"]
    pdb_path = tmp_path / "dcm.pdb"
    write_monomer_pdb_for_packmol(
        pdb_path, coords, Z, atom_names=names, resname="DCM"
    )
    text = pdb_path.read_text()
    assert " DCM " in text or "DCM A" in text
    assert "UNK" not in text
    assert "CL1 DCM" in text or "CL1 DCM" in text.replace("  ", " ")
    for line in text.splitlines():
        if "CL1" in line or "CL2" in line:
            assert line.rstrip().endswith("Cl")


def test_run_packmol_sphere_mixed_writes_inp(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_placement

    captured: dict[str, str] = {}

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input
        captured["inp_path"] = str(inp_path)

    monkeypatch.setattr(packmol_placement, "execute_packmol_script", fake_execute)

    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_a.write_text("END\n")
    pdb_b.write_text("END\n")
    out = tmp_path / "cluster.pdb"

    packmol_placement.run_packmol_sphere_mixed(
        [(pdb_a, 3), (pdb_b, 2)],
        center=(1.0, 2.0, 3.0),
        radius=20.0,
        output_pdb=out,
        tolerance=1.5,
        seed=42,
    )

    assert "inside sphere 1.0 2.0 3.0 20.0" in captured["input"]
    assert "number 3" in captured["input"]
    assert "number 2" in captured["input"]
    assert "tolerance 1.5" in captured["input"]
    assert "seed 42" in captured["input"]
    assert str(out) in captured["input"]


def test_run_packmol_cube_mixed_writes_inp(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_placement

    captured: dict[str, str] = {}

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input

    monkeypatch.setattr(packmol_placement, "execute_packmol_script", fake_execute)

    pdb_a = tmp_path / "a.pdb"
    pdb_a.write_text("END\n")
    out = tmp_path / "cluster.pdb"

    packmol_placement.run_packmol_cube_mixed(
        [(pdb_a, 5)],
        center=(0.0, 0.0, 0.0),
        cube_side=40.0,
        output_pdb=out,
        tolerance=2.0,
        seed=7,
    )

    assert "inside cube -20.0 -20.0 -20.0 40.0" in captured["input"]
    assert "number 5" in captured["input"]
    assert "seed 7" in captured["input"]
