"""Shared ML/MM handoff taper — single source of truth for calculator + training."""

from __future__ import annotations

import numpy as np
import pytest


def _fn():
    from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale

    return ml_switch_scale


# Defaults reported by the MD Calculator Summary:
#   ml_switch_width 1.5, mm_switch_on 8.0
#   ML fully-on 0 -> 6.5 ; handoff 6.5 -> 8.0
ON, WIDTH = 8.0, 1.5


def test_ml_fully_on_inside_handoff_start():
    f = _fn()
    for r in (0.0, 3.0, 6.0, 6.4999):
        assert float(f(np.float64(r), mm_switch_on=ON, ml_switch_width=WIDTH)) == pytest.approx(1.0)


def test_ml_fully_off_at_and_beyond_mm_switch_on():
    f = _fn()
    for r in (8.0, 9.0, 13.0, 50.0):
        assert float(f(np.float64(r), mm_switch_on=ON, ml_switch_width=WIDTH)) == pytest.approx(0.0)


def test_handoff_region_is_monotonic_decreasing_in_0_1():
    f = _fn()
    rs = np.linspace(6.5, 8.0, 25)
    vals = np.array([float(f(np.float64(r), mm_switch_on=ON, ml_switch_width=WIDTH)) for r in rs])
    assert vals[0] == pytest.approx(1.0)
    assert vals[-1] == pytest.approx(0.0)
    assert np.all(np.diff(vals) <= 1e-12)          # monotonic non-increasing
    assert np.all((vals >= -1e-12) & (vals <= 1 + 1e-12))


def test_handoff_start_follows_switch_width():
    """Fully-on edge is mm_switch_on - ml_switch_width, not a hardcoded 6.5."""
    f = _fn()
    # width 5.0 -> fully on below 3.0
    assert float(f(np.float64(2.9), mm_switch_on=ON, ml_switch_width=5.0)) == pytest.approx(1.0)
    assert float(f(np.float64(7.0), mm_switch_on=ON, ml_switch_width=5.0)) < 1.0


def test_is_differentiable_and_vmappable():
    import jax
    import jax.numpy as jnp

    f = _fn()
    g = jax.grad(lambda r: f(r, mm_switch_on=ON, ml_switch_width=WIDTH))
    # zero gradient in the flat regions, non-zero inside the handoff
    assert float(g(jnp.float64(3.0))) == pytest.approx(0.0)
    assert float(g(jnp.float64(9.0))) == pytest.approx(0.0)
    assert abs(float(g(jnp.float64(7.25)))) > 0.0
    out = jax.vmap(lambda r: f(r, mm_switch_on=ON, ml_switch_width=WIDTH))(jnp.linspace(0.0, 13.0, 8))
    assert out.shape == (8,)


def test_switch_gradient_matches_finite_difference():
    """The taper derivative contributes to forces in the handoff region."""
    import jax
    import jax.numpy as jnp

    f = lambda r: _fn()(r, mm_switch_on=ON, ml_switch_width=WIDTH)
    for r in (6.7, 7.0, 7.25, 7.8):
        h = 1e-3
        numerical = (float(f(r + h)) - float(f(r - h))) / (2 * h)
        assert float(jax.grad(f)(jnp.asarray(r))) == pytest.approx(numerical, rel=2e-3, abs=1e-4)
