"""ADUMB traced-RC preflight vs umbrella max (UM1RXN guard)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.restraints import (
    AdumbRcGuard,
    adumb_rc_wall_droff,
    check_adumb_rc_before_overlap_chunk,
    measure_adumb_rc_distances,
)


def test_adumb_rc_wall_droff_below_max() -> None:
    assert adumb_rc_wall_droff(8.0, margin=0.75) == pytest.approx(7.25)


def test_measure_adumb_rc_distances_from_coords() -> None:
    x = np.array([0.0, 3.0, 0.0], dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._unique_atom_index_by_name",
        side_effect=[1, 2],
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints._positions_xyz",
        return_value=(x, y, z),
    ):
        dists = measure_adumb_rc_distances((("CL1", "C1"),))
    assert dists["CL1-C1"] == pytest.approx(3.0)


def test_check_adumb_rc_raises_at_hard_limit() -> None:
    guard = AdumbRcGuard(rcmax=8.0, rcwall=500.0, pairs=(("CL1", "C1"),))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.restraints.measure_adumb_rc_distances",
        return_value={"CL1-C1": 8.01},
    ):
        with pytest.raises(RuntimeError, match="UM1RXN would abort"):
            check_adumb_rc_before_overlap_chunk(
                guard,
                overlap_context="HEAT",
                chunk_index=43,
                n_chunks=400,
            )
