"""Unit tests for the HEAT order-of-magnitude hybrid-GRMS abort decision."""

from __future__ import annotations

import numpy as np


def _fn():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        heat_grms_jump_abort_reason,
    )

    return heat_grms_jump_abort_reason


def test_aborts_on_order_of_magnitude_jump_above_floor():
    # The reported failure: ~22 baseline -> ~322 kcal/mol/Å.
    reason = _fn()(22.34, 322.0)
    assert reason is not None
    assert "jumped" in reason


def test_no_abort_when_below_absolute_floor_even_if_high_ratio():
    # 1 -> 140: ratio 140x but current below the 150 floor -> thermostat noise.
    assert _fn()(1.0, 140.0) is None
    # Just over the floor with a >=10x ratio does abort.
    assert _fn()(1.0, 200.0) is not None


def test_no_abort_when_ratio_below_factor():
    # 22.34 -> 200: above floor but only ~9x (< 10x) -> no abort.
    assert _fn()(22.34, 200.0) is None
    # 22.34 -> 250: >=10x and above floor -> abort.
    assert _fn()(22.34, 250.0) is not None


def test_custom_factor_and_floor():
    assert _fn()(10.0, 45.0, factor=4.0, floor_kcalmol_A=40.0) is not None
    assert _fn()(10.0, 45.0, factor=5.0, floor_kcalmol_A=40.0) is None
    assert _fn()(10.0, 35.0, factor=4.0, floor_kcalmol_A=40.0) is None


def test_none_and_nonfinite_are_safe():
    fn = _fn()
    assert fn(None, 500.0) is None
    assert fn(22.0, None) is None
    assert fn(np.nan, 500.0) is None
    assert fn(22.0, np.nan) is None
    assert fn(0.0, 500.0) is None
