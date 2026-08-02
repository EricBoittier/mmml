"""Tests for PSF residue selection / subset writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.utils.psf_subset import (
    indices_for_resnames,
    parse_resname_list,
    write_subset_psf,
)


def _tiny_psf(path: Path) -> None:
    # 1 TRIA (2 atoms) + 1 TIP3 (3 atoms)
    path.write_text(
        "\n".join(
            [
                "PSF",
                "",
                "       5 !NATOM",
                "       1 A    1    TRIA N    N     -0.470000       14.0070           0",
                "       2 A    1    TRIA HN   H      0.310000        1.0080           0",
                "       3 WAT  1    TIP3 OH2  OT    -0.834000       15.9994           0",
                "       4 WAT  1    TIP3 H1   HT     0.417000        1.0080           0",
                "       5 WAT  1    TIP3 H2   HT     0.417000        1.0080           0",
                "",
                "       3 !NBOND: bonds",
                "       1       2       3       4       3       5",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_resname_list():
    assert parse_resname_list("tria, tip3") == ["TRIA", "TIP3"]
    assert parse_resname_list(["TRIA", "TRIA"]) == ["TRIA"]


def test_indices_and_subset_psf(tmp_path: Path):
    psf = tmp_path / "model.psf"
    _tiny_psf(psf)
    idx, atoms = indices_for_resnames(psf, "TRIA")
    assert list(idx) == [0, 1]
    assert {a.resname for a in atoms} == {"TRIA"}

    out = tmp_path / "tria.psf"
    write_subset_psf(psf, out, idx)
    text = out.read_text(encoding="utf-8")
    assert "2 !NATOM" in text or "       2 !NATOM" in text
    assert "TRIA" in text
    assert "TIP3" not in text
    assert "1 !NBOND" in text or "       1 !NBOND" in text
