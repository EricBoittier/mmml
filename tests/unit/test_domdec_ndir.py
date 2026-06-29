"""Tests for CHARMM DOMDEC NDIR selection."""

from __future__ import annotations

import pytest

from mmml.utils.domdec_ndir import format_domdec_ndir, suggest_domdec_ndir


@pytest.mark.parametrize(
    ("n_ranks", "expected"),
    [
        (1, (1, 1, 1)),
        (2, (2, 1, 1)),
        (4, (4, 1, 1)),
        (8, (8, 1, 1)),
    ],
)
def test_suggest_domdec_ndir_avoids_y_split_2_to_7(n_ranks: int, expected: tuple[int, int, int]) -> None:
    assert suggest_domdec_ndir(n_ranks) == expected
    parts = [int(x) for x in format_domdec_ndir(n_ranks).split()]
    assert parts[1] in (1, 8) or parts[1] >= 8
    assert parts[0] * parts[1] * parts[2] == n_ranks
