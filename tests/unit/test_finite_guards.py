"""Padding sanitization vs nonfinite physical contributions."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.finite_guards import (
    NonfinitePhysicalContribution,
    require_host_finite,
)


def test_require_host_finite_accepts_zeros():
    require_host_finite(0.0, np.zeros((3, 3)))


def test_require_host_finite_rejects_nan_energy():
    with pytest.raises(NonfinitePhysicalContribution) as exc:
        require_host_finite(np.nan, np.zeros((2, 3)), name="ML USER")
    assert "ML USER" in str(exc.value)
    assert "energy_bad=1" in str(exc.value)


def test_require_host_finite_rejects_inf_force():
    f = np.zeros((2, 3))
    f[1, 0] = np.inf
    with pytest.raises(NonfinitePhysicalContribution):
        require_host_finite(1.0, f)
