"""Tests for CHARMM DOMDEC NDIR selection."""

from __future__ import annotations

import pytest

from mmml.utils.domdec_ndir import (
    format_domdec_ndir,
    min_domdec_crystal_side_A,
    min_domdec_mpi_ranks,
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


def test_min_domdec_mpi_ranks() -> None:
    assert min_domdec_mpi_ranks() == 1
    assert min_domdec_mpi_ranks(allow_serial=False) == 8
