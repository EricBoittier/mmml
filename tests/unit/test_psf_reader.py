"""Unit tests for PSF atom type reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.utils.psf_reader import read_psf_atom_types

REPO = Path(__file__).resolve().parents[2]


def test_read_psf_atom_types_counts():
    psf = REPO / "artifacts/pbc_liquid_density_dyn_smoke/dcm_32/pycharmm_init/model.psf"
    if not psf.is_file():
        pytest.skip("smoke PSF missing")
    types = read_psf_atom_types(psf)
    assert types.dtype.kind == "U"
    assert types.shape == (160,)
    uniq, counts = np.unique(types, return_counts=True)
    assert set(uniq.tolist()) == {"CG321", "HGA2", "CLGA1"}
    assert int(counts.sum()) == 160


def test_read_psf_atom_types_missing_natom(tmp_path: Path):
    psf = tmp_path / "bad.psf"
    psf.write_text("* title\n", encoding="utf-8")
    with pytest.raises(ValueError, match="!NATOM"):
        read_psf_atom_types(psf)


def test_read_psf_atom_types_malformed_line(tmp_path: Path):
    psf = tmp_path / "bad.psf"
    psf.write_text("2 !NATOM\n1 SEG RES 1 ALA CA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before EOF"):
        read_psf_atom_types(psf)
