"""Unit tests for Packmol-backed monomer repack during MD recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _two_monomer_system() -> tuple[np.ndarray, np.ndarray]:
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    return pos, offsets


def test_packmol_selective_repack_writes_fixed_and_movable_blocks(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_repack

    captured: dict[str, str] = {}
    pos, offsets = _two_monomer_system()
    z = np.array([6, 1, 6, 1], dtype=int)

    def fake_execute(packmol_input: str, inp_path: Path) -> None:
        captured["input"] = packmol_input
        captured["inp_path"] = str(inp_path)
        out = tmp_path / "repack.pdb"
        out.write_text(
            "\n".join(
                [
                    "ATOM      1  C   F000A   1       0.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      2  H1  F000A   1       1.000   0.000   0.000  1.00  0.00           H",
                    "ATOM      3  C   M001A   2       5.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      4  H1  M001A   2       6.000   0.000   0.000  1.00  0.00           H",
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", fake_execute)
    monkeypatch.setattr(
        packmol_repack,
        "_charmm_atom_metadata",
        lambda _n: (["C", "H1", "C", "H1"], z),
    )

    out = packmol_repack.repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [1],
        min_distance=1.5,
        spacing=4.0,
        seed=7,
        cell=np.diag([10.0, 10.0, 10.0]),
        scratch_dir=tmp_path,
        atomic_numbers=z,
    )

    assert "fixed 0.5 0.0 0.0 0. 0. 0." in captured["input"]
    assert "number 1" in captured["input"]
    assert "number 1\n  inside cube" in captured["input"] or "inside cube" in captured["input"]
    assert "seed 7" in captured["input"]
    assert out[2, 0] == pytest.approx(5.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_read_packmol_monomer_coords_splits_sequential_atoms_not_residue_ids(tmp_path):
    from mmml.interfaces.pycharmmInterface.packmol_repack import (
        _read_packmol_monomer_coords,
    )

    out = tmp_path / "repack.pdb"
    # Three 2-atom monomers; Packmol often leaves all fixed atoms on residue 1.
    lines = []
    serial = 1
    for atom_idx in range(6):
        x = float(atom_idx + 1)
        lines.append(
            f"ATOM      {serial}  C   FIX A   1       {x:.3f}   0.000   0.000  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    packed = _read_packmol_monomer_coords(out, expected_atoms_per_monomer=[2, 2, 2])
    assert len(packed) == 3
    assert packed[0].shape == (2, 3)
    assert packed[1][0, 0] == pytest.approx(3.0)
    assert packed[2][1, 0] == pytest.approx(6.0)


def test_packmol_repack_falls_back_to_grid_when_binary_missing(monkeypatch):
    from mmml.interfaces.pycharmmInterface import packmol_repack

    pos, offsets = _two_monomer_system()
    z = np.array([6, 1, 6, 1], dtype=int)

    def boom(*_a, **_k):
        raise FileNotFoundError("packmol missing")

    monkeypatch.setattr(packmol_repack, "execute_packmol_script", boom)
    monkeypatch.setattr(
        packmol_repack,
        "_charmm_atom_metadata",
        lambda _n: (["C", "H1", "C", "H1"], z),
    )

    out = packmol_repack.repack_selected_monomers_clear_overlap(
        pos,
        offsets,
        [1],
        min_distance=1.5,
        spacing=4.0,
        seed=11,
        cell=None,
        atomic_numbers=z,
    )
    assert out.shape == pos.shape
    assert np.linalg.norm(out[2:] - pos[2:]) > 0.1


def test_overlap_guard_repack_fn_uses_packmol_module():
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        _repack_monomers_clear_overlap_fn,
    )
    from mmml.interfaces.pycharmmInterface import packmol_repack

    assert _repack_monomers_clear_overlap_fn() is packmol_repack.repack_monomers_clear_overlap
