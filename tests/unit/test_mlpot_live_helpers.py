"""Pure helpers used by live MLpot optimizer/dynamics tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.functionality.mlpot._live_helpers import max_displacement, subset_positions


def test_max_displacement_zero_for_identical_coords() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert max_displacement(pos, pos) == 0.0


def test_max_displacement_reports_largest_atom_delta() -> None:
    a = np.zeros((2, 3))
    b = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]])
    assert max_displacement(a, b) == pytest.approx(0.5)


def test_subset_positions_uses_one_based_charmm_indexes() -> None:
    pos = np.arange(12, dtype=float).reshape(4, 3)
    out = subset_positions(pos, [2, 4])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[0], pos[1])
    np.testing.assert_allclose(out[1], pos[3])
