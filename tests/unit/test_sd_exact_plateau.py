"""Unit tests for exact CHARMM-SD plateau detection."""

from __future__ import annotations

import numpy as np


def _fn():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import is_exact_sd_plateau

    return is_exact_sd_plateau


def test_identical_grms_is_plateau():
    # CHARMM SD reporting the same GRMS chunk after chunk.
    assert _fn()(37.74, 37.74) is True


def test_tiny_float_noise_still_plateau():
    assert _fn()(37.74, 37.7401) is True


def test_real_progress_is_not_plateau():
    assert _fn()(37.74, 30.0) is False
    assert _fn()(9.02, 8.55) is False


def test_none_and_nonfinite_safe():
    fn = _fn()
    assert fn(None, 10.0) is False
    assert fn(10.0, None) is False
    assert fn(np.nan, 10.0) is False
    assert fn(10.0, np.inf) is False


def test_custom_rel_tol():
    assert _fn()(100.0, 100.4, rel_tol=1e-2) is True
    assert _fn()(100.0, 102.0, rel_tol=1e-2) is False
