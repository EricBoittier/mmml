"""Tests for CHARMM DOMDEC NDIR selection.

Key confirmed fact (c47 runtime, 2026-06-29):
  domdec.F90 enforces "1 or >=8 nodes per axis" for ALL c47 builds.
  With np=2, every possible NDIR split violates this → DOMDEC requires np>=8.
  strict_c47_axis_rule=True is therefore the default in all public functions.
"""

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


# ---------------------------------------------------------------------------
# Default (strict_c47_axis_rule=True) — ALL c47 builds enforce 1 or >=8/axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_ranks", [2, 3, 4, 5, 6, 7])
def test_suggest_domdec_ndir_rejects_small_np_by_default(n_ranks: int) -> None:
    """np < 8 raises ValueError by default (strict_c47_axis_rule=True)."""
    with pytest.raises(ValueError, match="n_ranks"):
        suggest_domdec_ndir(n_ranks)


@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (8, (8, 1, 1)),
        (16, (16, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_valid_np(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks) == expected
    assert suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=True) == expected


# ---------------------------------------------------------------------------
# Non-strict mode (custom non-c47 builds, opt-in only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (1, (1, 1, 1)),
        (2, (2, 1, 1)),
        (4, (4, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_non_strict_small_np(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=False) == expected


# ---------------------------------------------------------------------------
# Crystal side minimums
# ---------------------------------------------------------------------------

def test_min_domdec_crystal_side_np2_non_strict() -> None:
    # Non-strict: np=2 → NDIR 2 1 1, domain = cutnb(15) + halo(4) = 19 Å per domain
    assert min_domdec_crystal_side_A(2, strict_c47_axis_rule=False) == 38.0


def test_min_domdec_crystal_side_np8_default() -> None:
    # Default strict: np=8 → NDIR 8 1 1, min box = 8 × 19 = 152 Å
    assert min_domdec_crystal_side_A(8) == 152.0


# ---------------------------------------------------------------------------
# min_domdec_mpi_ranks
# ---------------------------------------------------------------------------

def test_min_domdec_mpi_ranks() -> None:
    assert min_domdec_mpi_ranks() == 1
    assert min_domdec_mpi_ranks(allow_serial=False) == 8          # strict default
    assert min_domdec_mpi_ranks(allow_serial=False, strict_c47_axis_rule=False) == 2


# ---------------------------------------------------------------------------
# prep dir picker
# ---------------------------------------------------------------------------

def test_pick_domdec_prep_dir_prefers_smallest_tier3_box(tmp_path: Path) -> None:
    small = tmp_path / "domdec_dcm10_l40"
    large = tmp_path / "domdec_dcm10_l152"
    for d in (small, large):
        d.mkdir()
        (d / "model.psf").write_text("stub")
        (d / "box.json").write_text(f'{{"box_side_A": {d.name.rsplit("_l", 1)[1]}.0}}')
    picked = pick_domdec_prep_dir(tmp_path, n_dcm=10, min_side_A=38.0, prefer_smallest=True)
    assert picked == small


# ---------------------------------------------------------------------------
# format_domdec_tier3_energy_block — strict default means np=2 raises
# ---------------------------------------------------------------------------

def test_format_domdec_tier3_energy_block_np2_raises_by_default() -> None:
    with pytest.raises(ValueError, match="n_ranks"):
        format_domdec_tier3_energy_block(2)


def test_format_domdec_tier3_energy_block_np2_non_strict() -> None:
    block = format_domdec_tier3_energy_block(2, strict_c47_axis_rule=False)
    assert "energy cutnb 15.0 cutim 15.0" in block
    assert "domd ndir 2 1 1 split off" in block
    assert block == format_domdec_charmm_commands(2, strict_c47_axis_rule=False)
    assert block.count("-") >= 2
    assert block.strip().splitlines()[-1].strip().startswith("domd ndir")


def test_format_domdec_tier3_energy_block_np8() -> None:
    block = format_domdec_tier3_energy_block(8)
    assert "domd ndir 8 1 1 split off" in block
