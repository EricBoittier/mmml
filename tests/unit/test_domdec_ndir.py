"""Tests for CHARMM DOMDEC NDIR selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmml.utils.domdec_ndir import (
    format_domdec_charmm_commands,
    format_domdec_ndir,
    format_domdec_tier3_energy_block,
    min_domdec_crystal_side_A,
    min_domdec_mpi_ranks,
    pick_domdec_prep_dir,
    suggest_domdec_ndir,
)


@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (1, (1, 1, 1)),
        (2, (2, 1, 1)),
        (4, (4, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_default_small_np(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks) == expected
    assert suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=False) == expected


@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (8, (8, 1, 1)),
        (16, (16, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_c47_strict(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=True) == expected


@pytest.mark.parametrize("n_ranks", [2, 3, 4, 5, 6, 7])
def test_suggest_domdec_ndir_rejects_small_np_on_c47(n_ranks: int) -> None:
    with pytest.raises(ValueError, match="n_ranks"):
        suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=True)


def test_min_domdec_crystal_side_np2_dense_box() -> None:
    assert min_domdec_crystal_side_A(2) == 38.0


def test_min_domdec_crystal_side_np8_c47() -> None:
    assert min_domdec_crystal_side_A(8, strict_c47_axis_rule=True) == 152.0


def test_pick_domdec_prep_dir_prefers_smallest_tier3_box(tmp_path: Path) -> None:
    small = tmp_path / "domdec_dcm10_l40"
    large = tmp_path / "domdec_dcm10_l152"
    for d in (small, large):
        d.mkdir()
        (d / "model.psf").write_text("stub")
        (d / "box.json").write_text(f'{{"box_side_A": {d.name.rsplit("_l", 1)[1]}.0}}')
    picked = pick_domdec_prep_dir(tmp_path, n_dcm=10, min_side_A=38.0, prefer_smallest=True)
    assert picked == small


def test_min_domdec_mpi_ranks() -> None:
    assert min_domdec_mpi_ranks() == 1
    assert min_domdec_mpi_ranks(allow_serial=False) == 2
    assert min_domdec_mpi_ranks(allow_serial=False, strict_c47_axis_rule=True) == 8


def test_format_domdec_charmm_commands_np2() -> None:
    block = format_domdec_tier3_energy_block(2)
    assert "energy cutnb 15.0" in block
    assert "domdec ndir 2 1 1" in block
    assert block == format_domdec_charmm_commands(2)
    assert block.count("-") >= 3
    # domdec.info Example 1: domdec ndir on continued energy line, not a separate command.
    assert block.strip().splitlines()[-1].strip().startswith("domdec ndir")
