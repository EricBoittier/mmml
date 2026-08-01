"""POV-Ray liquid-box renders: orthographic camera + cell when present."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_liquid_box_povray.py"
_SPEC = importlib.util.spec_from_file_location("render_liquid_box_povray", _SCRIPT)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(mod)


def _water(*, cell: float | None = None) -> Atoms:
    atoms = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    if cell is not None:
        atoms.set_cell([cell, cell, cell])
        atoms.set_pbc(True)
    return atoms


def test_pov_uses_orthographic_camera(tmp_path: Path):
    pov = tmp_path / "box.pov"
    mod.write_liquid_box_pov(_water(cell=10.0), pov, width=200)
    text = pov.read_text()
    assert "camera {orthographic" in text
    assert "perspective" not in text


def test_pov_draws_cell_edges_when_cell_present(tmp_path: Path):
    pov = tmp_path / "box.pov"
    drawn = mod.write_liquid_box_pov(_water(cell=28.0), pov, width=200)
    assert drawn
    text = pov.read_text()
    assert "#declare Rcell = 0.050;" in text
    # ASE emits 12 edge cylinders for a parallelepiped.
    assert text.count("Rcell pigment {Black}") == 12


def test_pov_omits_cell_when_absent(tmp_path: Path):
    pov = tmp_path / "box.pov"
    drawn = mod.write_liquid_box_pov(_water(cell=None), pov, width=200)
    assert not drawn
    text = pov.read_text()
    assert "camera {orthographic" in text
    assert "// no cell vertices" in text
    assert "Rcell pigment {Black}" not in text


def test_attach_cell_from_box_json(tmp_path: Path):
    struct = tmp_path / "model.pdb"
    _water().write(struct)
    (tmp_path / "box.json").write_text(json.dumps({"box_side_A": 28.0}))
    side = mod.resolve_box_side_A(struct)
    assert side == pytest.approx(28.0)
    atoms, attached = mod.attach_cell_if_needed(_water(), side)
    assert attached
    assert atoms.cell.rank == 3
    np.testing.assert_allclose(atoms.cell.lengths(), [28.0, 28.0, 28.0])
    # Centroid lands at the cube centre so the wireframe encloses the liquid.
    np.testing.assert_allclose(atoms.get_positions().mean(axis=0), [14.0, 14.0, 14.0], atol=1e-9)


def test_box_side_cli_overrides_json(tmp_path: Path):
    struct = tmp_path / "model.pdb"
    _water().write(struct)
    (tmp_path / "box.json").write_text(json.dumps({"box_side_A": 28.0}))
    side = mod.resolve_box_side_A(struct, box_side=30.0)
    assert side == pytest.approx(30.0)


def test_existing_structure_cell_is_not_overwritten():
    atoms, attached = mod.attach_cell_if_needed(_water(cell=12.0), 28.0)
    assert not attached
    np.testing.assert_allclose(atoms.cell.lengths(), [12.0, 12.0, 12.0])
