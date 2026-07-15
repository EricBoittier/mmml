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


def _temp_fn():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        heat_temperature_abort_reason,
    )

    return heat_temperature_abort_reason


def test_temp_abort_low_target_runaway():
    fn = _temp_fn()
    # 50 K target, 1027 K live (the observed early-chunk overshoot) -> abort.
    assert fn(50.0, 1027.0) is not None
    # 50 K target, 300 K live -> below the 400 K absolute ceiling -> no abort.
    assert fn(50.0, 300.0) is None
    # Just over the absolute ceiling.
    assert fn(50.0, 450.0) is not None


def test_temp_abort_scales_with_high_target():
    fn = _temp_fn()
    # 300 K target: ceiling = 8*300 = 2400 K.
    assert fn(300.0, 1000.0) is None
    assert fn(300.0, 3000.0) is not None


def test_temp_abort_unknown_target_uses_absolute_ceiling():
    fn = _temp_fn()
    assert fn(None, 500.0) is not None   # > 400 K floor
    assert fn(None, 300.0) is None


def test_temp_abort_nonfinite_safe():
    import numpy as np

    fn = _temp_fn()
    assert fn(50.0, None) is None
    assert fn(50.0, np.nan) is None


def test_temp_abort_custom_factor():
    fn = _temp_fn()
    # factor 20 at 50 K target -> ceiling 1000 K; 900 K stays under.
    assert fn(50.0, 900.0, factor=20.0) is None
    assert fn(50.0, 1100.0, factor=20.0) is not None
