"""Packmol truncates 5-char CGenFF names; rewrite restores them from templates."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_template(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rewrite_packmol_pdb_resnames_restores_ch3cl(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        rewrite_packmol_pdb_resnames,
    )

    solute = tmp_path / "initial.pdb"
    tip3 = tmp_path / "tip3.pdb"
    packed = tmp_path / "init-tip3box.pdb"

    # Chain-less CGenFF layout (same as examples/m export).
    _write_template(
        solute,
        [
            "REMARK solute",
            "ATOM      1  N1  AMM1    1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  H11 AMM1    1       1.000   0.000   0.000  1.00  0.00           H",
            "ATOM      3  C1  CH3CL   2       2.000   0.000   0.000  1.00  0.00           C",
            "ATOM      4  CL1 CH3CL   2       3.000   0.000   0.000  1.00  0.00          Cl",
            "END",
        ],
    )
    _write_template(
        tip3,
        [
            "REMARK tip3",
            "ATOM      1  OH2 TIP3    1       0.000   0.000   0.000  1.00  0.00           O",
            "ATOM      2  H1  TIP3    1       1.000   0.000   0.000  1.00  0.00           H",
            "ATOM      3  H2  TIP3    1       0.000   1.000   0.000  1.00  0.00           H",
            "END",
        ],
    )
    # Simulate Packmol 20.x output: chain embedded, CH3CL → CH3CA.
    packed_lines = [
        "REMARK packmol mangled",
        "ATOM      1  N1  AMM1A   1      10.000  10.000  10.000  1.00  0.00           N",
        "ATOM      2  H11 AMM1A   1      11.000  10.000  10.000  1.00  0.00           H",
        "ATOM      3  C1  CH3CA   2      12.000  10.000  10.000  1.00  0.00           C",
        "ATOM      4  CL1 CH3CA   2      13.000  10.000  10.000  1.00  0.00          Cl",
        "ATOM      5  OH2 TIP3B   3      14.000  10.000  10.000  1.00  0.00           O",
        "ATOM      6  H1  TIP3B   3      15.000  10.000  10.000  1.00  0.00           H",
        "ATOM      7  H2  TIP3B   3      14.000  11.000  10.000  1.00  0.00           H",
        "END",
    ]
    packed.write_text("\n".join(packed_lines) + "\n", encoding="utf-8")

    rewrite_packmol_pdb_resnames(packed, [(solute, 1), (tip3, 1)])
    text = packed.read_text(encoding="utf-8")
    atom_lines = [ln for ln in text.splitlines() if ln.startswith("ATOM")]
    assert len(atom_lines) == 7
    assert sum("AMM1" in ln and "AMM1A" not in ln for ln in atom_lines) == 2
    assert sum("CH3CL" in ln for ln in atom_lines) == 2
    assert sum("TIP3" in ln and "TIP3B" not in ln for ln in atom_lines) == 3
    assert "CH3CA" not in text
    # Coords preserved from Packmol output.
    assert "10.000" in atom_lines[0]
    assert "13.000" in atom_lines[3]
    for ln in atom_lines:
        assert ln[22:26].strip().isdigit(), ln
        float(ln[30:38])
        float(ln[38:46])
        float(ln[46:54])

    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _residue_sequence_from_pdb,
    )

    assert _residue_sequence_from_pdb(packed) == ["AMM1", "CH3CL", "TIP3"]
    # Positions still finite / ordered.
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        _parse_pdb_atom_records,
    )

    _n, _r, pos = _parse_pdb_atom_records(packed)
    assert pos.shape == (7, 3)
    assert np.allclose(pos[0], [10.0, 10.0, 10.0])
