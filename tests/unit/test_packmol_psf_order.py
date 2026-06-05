"""Unit tests for Packmol PDB -> PSF coordinate reordering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_packmol_like_pdb(
    path: Path,
    *,
    records: list[tuple[int, str, str, tuple[float, float, float]]],
) -> None:
    """Write minimal ATOM records (serial, resid, name, xyz)."""
    lines = [
        "REMARK   mmml test packmol output",
        "CRYST1   200.000   200.000   200.000  90.00  90.00  90.00 P 1           1",
    ]
    for serial, resid, name, (x, y, z) in records:
        lines.append(
            f"ATOM  {serial:5d} {name[:4]:>4s} DCM A{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_assign_packmol_pdb_to_psf_order_reorders_by_resid_and_name(tmp_path):
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    psf_names = ["C1", "H1", "C2", "H2"]
    atoms_per_list = [2, 2]
    pdb_path = tmp_path / "cluster.pdb"
    _write_packmol_like_pdb(
        pdb_path,
        records=[
            (1, 2, "C2", (3.0, 0.0, 1.0)),
            (2, 1, "H1", (0.0, 1.0, 0.5)),
            (3, 2, "H2", (3.0, 1.0, 0.5)),
            (4, 1, "C1", (0.0, 0.0, 0.0)),
        ],
    )

    out = assign_packmol_pdb_to_psf_order(pdb_path, psf_names, atoms_per_list)

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [3.0, 0.0, 1.0],
            [3.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(out, expected)


def test_assign_packmol_pdb_to_psf_order_rejects_mismatched_keys(tmp_path):
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    pdb_path = tmp_path / "cluster.pdb"
    _write_packmol_like_pdb(
        pdb_path,
        records=[
            (1, 1, "C1", (0.0, 0.0, 0.0)),
            (2, 1, "XX", (1.0, 0.0, 0.0)),
        ],
    )

    with pytest.raises(RuntimeError, match="does not match PSF atom order"):
        assign_packmol_pdb_to_psf_order(pdb_path, ["C1", "H1"], [2])


def test_assign_packmol_pdb_to_psf_order_rejects_flat_cluster(tmp_path):
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    pdb_path = tmp_path / "flat.pdb"
    _write_packmol_like_pdb(
        pdb_path,
        records=[
            (1, 1, "C1", (0.0, 0.0, 0.0)),
            (2, 2, "C2", (5.0, 0.0, 0.0)),
        ],
    )

    with pytest.raises(RuntimeError, match="Packmol cluster not 3D"):
        assign_packmol_pdb_to_psf_order(pdb_path, ["C1", "C2"], [1, 1])
