"""Tests for CHARMM DOMDEC NDIR selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.utils.domdec_ndir import (
    format_domdec_ndir,
    min_domdec_crystal_side_A,
    min_domdec_mpi_ranks,
    pick_domdec_prep_dir,
    suggest_domdec_ndir,
)


@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (1, (1, 1, 1)),
        (8, (8, 1, 1)),
        (16, (16, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_c47_axis_rule(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks) == expected
    nx, ny, nz = (int(x) for x in format_domdec_ndir(n_ranks).split())
    for axis in (nx, ny, nz):
        assert axis == 1 or axis >= 8
    assert nx * ny * nz == n_ranks


@pytest.mark.parametrize("n_ranks", [2, 3, 4, 5, 6, 7])
def test_suggest_domdec_ndir_rejects_small_np(n_ranks: int) -> None:
    with pytest.raises(ValueError, match="n_ranks"):
        suggest_domdec_ndir(n_ranks)


def test_min_domdec_crystal_side_for_np8() -> None:
    assert min_domdec_crystal_side_A(8) == 152.0


def test_pick_domdec_prep_dir_prefers_smallest_tier3_box(tmp_path: Path) -> None:
    small = tmp_path / "domdec_dcm10_l40"
    large = tmp_path / "domdec_dcm10_l152"
    for d in (small, large):
        d.mkdir()
        (d / "model.psf").write_text("stub")
        (d / "box.json").write_text(f'{{"box_side_A": {d.name.rsplit("_l", 1)[1]}.0}}')
    picked = pick_domdec_prep_dir(tmp_path, n_dcm=10, min_side_A=152.0)
    assert picked == large


def test_pick_domdec_prep_dir_newest_for_validate(tmp_path: Path) -> None:
    a = tmp_path / "domdec_dcm10_l40"
    b = tmp_path / "domdec_dcm10_l152"
    for d in (a, b):
        d.mkdir()
        (d / "model.psf").write_text("stub")
    picked = pick_domdec_prep_dir(tmp_path, n_dcm=10, min_side_A=0.0)
    assert picked in {a, b}


def test_min_domdec_mpi_ranks() -> None:
    assert min_domdec_mpi_ranks() == 1
    assert min_domdec_mpi_ranks(allow_serial=False) == 8
