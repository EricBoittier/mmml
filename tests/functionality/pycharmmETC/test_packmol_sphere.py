"""Tests for spherical Packmol helpers (no Packmol binary required)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_packmol_sphere_use_defaults():
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_sphere_use

    assert resolve_packmol_sphere_use(composition="MEOH:5", flat_bottom_radius=12.0) is True
    assert resolve_packmol_sphere_use(composition="MEOH:5", flat_bottom_radius=None) is False
    assert (
        resolve_packmol_sphere_use(
            composition="MEOH:5",
            flat_bottom_radius=12.0,
            packmol_sphere=False,
        )
        is False
    )
    assert (
        resolve_packmol_sphere_use(
            composition=None,
            flat_bottom_radius=12.0,
            packmol_sphere=True,
        )
        is True
    )


def test_require_packmol_sphere_radius():
    from mmml.interfaces.pycharmmInterface.packmol_placement import require_packmol_sphere_radius

    assert require_packmol_sphere_radius(15.5) == 15.5
    with pytest.raises(ValueError, match="flat-bottom-radius"):
        require_packmol_sphere_radius(None)


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
