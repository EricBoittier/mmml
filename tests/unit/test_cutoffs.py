"""Unit tests for ML/MM :class:`CutoffParameters` and switching scales."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.calculator_utils import (
    GAMMA_ON as CALC_GAMMA_ON,
    _sharpstep as calc_sharpstep,
    ml_switch_simple,
    mm_switch_simple,
)
from mmml.interfaces.pycharmmInterface.cutoffs import (
    GAMMA_OFF,
    GAMMA_ON,
    CutoffParameters,
    handoff_widths_from_args,
)


@pytest.fixture
def default_cp() -> CutoffParameters:
  return CutoffParameters(ml_switch_width=2.0, mm_switch_on=5.0, mm_switch_width=1.0)


def test_cutoff_defaults() -> None:
    cp = CutoffParameters()
    assert cp.ml_switch_width == 1.5
    assert cp.mm_switch_on == 8.0
    assert cp.mm_switch_width == 5.0
    assert cp.complementary_handoff is True


def test_cutoff_fixture_values(default_cp: CutoffParameters) -> None:
    assert default_cp.ml_switch_width == 2.0
    assert default_cp.mm_switch_on == 5.0
    assert default_cp.mm_switch_width == 1.0


def test_cutoff_deprecated_aliases_resolve() -> None:
    with pytest.warns(DeprecationWarning):
        cp = CutoffParameters(ml_cutoff=2.5, mm_cutoff=1.5, mm_switch_on=6.0)
    assert cp.ml_switch_width == 2.5
    assert cp.mm_switch_width == 1.5
    assert cp.ml_cutoff == 2.5
    assert cp.mm_cutoff == 1.5


def test_cutoff_eq_hash_and_roundtrip(default_cp: CutoffParameters) -> None:
    other = CutoffParameters.from_dict(default_cp.to_dict())
    assert default_cp == other
    assert hash(default_cp) == hash(other)
    assert default_cp != CutoffParameters(mm_switch_on=4.0)


def test_handoff_widths_from_args_namespace() -> None:
    class Args:
        ml_switch_width = 2.0
        mm_switch_on = 5.0
        mm_switch_width = 1.0

    assert handoff_widths_from_args(Args()) == (2.0, 5.0, 1.0)

    class Legacy:
        ml_cutoff = 2.5
        mm_switch_on = 5.0
        mm_cutoff = 0.8

    assert handoff_widths_from_args(Legacy()) == (2.5, 5.0, 0.8)


def test_ml_scale_matches_calculator_sharpstep(default_cp: CutoffParameters) -> None:
    """CutoffParameters.ml_scale must match the JAX calculator sharpstep definition."""
    r = np.linspace(0.0, 8.0, 50)
    start = default_cp.mm_switch_on - default_cp.ml_switch_width
    stop = default_cp.mm_switch_on
    expected = np.asarray(
        1.0 - calc_sharpstep(r, start, stop, gamma=GAMMA_ON), dtype=np.float64
    )
    np.testing.assert_allclose(
        default_cp.ml_scale(r, gamma_ml=GAMMA_ON),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.unit
def test_complementary_scales_match_jax_sharpstep(default_cp: CutoffParameters) -> None:
    """NumPy CutoffParameters scales match JAX _sharpstep (MM/jax-md switching path)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    r = jnp.linspace(0.01, 8.0, 60)
    start = float(default_cp.mm_switch_on - default_cp.ml_switch_width)
    stop = float(default_cp.mm_switch_on)
    tail = float(default_cp.mm_switch_on + default_cp.mm_switch_width)

    s_ml_jax = 1.0 - calc_sharpstep(r, start, stop, gamma=GAMMA_ON)
    handoff = 1.0 - s_ml_jax
    mm_taper = 1.0 - calc_sharpstep(r, stop, tail, gamma=GAMMA_OFF)
    s_mm_jax = handoff * mm_taper

    r_np = np.asarray(r, dtype=np.float64)
    s_ml_np, s_mm_np = default_cp.ml_mm_scales_complementary(
        r_np, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF
    )
    np.testing.assert_allclose(s_ml_np, np.asarray(s_ml_jax), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(s_mm_np, np.asarray(s_mm_jax), rtol=1e-6, atol=1e-6)
    # Guard against accidental no-op in the JAX path
    assert float(jax.device_get(jnp.max(s_ml_jax))) == pytest.approx(1.0)


def test_ml_scale_endpoint_values(default_cp: CutoffParameters) -> None:
    r0 = default_cp.mm_switch_on - default_cp.ml_switch_width
    assert default_cp.ml_scale(r0 - 0.01, gamma_ml=GAMMA_ON) == pytest.approx(1.0)
    assert default_cp.ml_scale(default_cp.mm_switch_on + 0.01, gamma_ml=GAMMA_ON) == pytest.approx(
        0.0, abs=1e-12
    )


def test_complementary_scales_sum_to_one_in_handoff(default_cp: CutoffParameters) -> None:
    """For r in [handoff_start, mm_switch_on] with MM taper still on, s_ML + s_MM = 1."""
    r0 = default_cp.mm_switch_on - default_cp.ml_switch_width
    r_vals = np.linspace(r0 + 0.05, default_cp.mm_switch_on - 0.05, 20)
    for r in r_vals:
        s_ml, s_mm = default_cp.ml_mm_scales_complementary(
            r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF
        )
        assert s_ml + s_mm == pytest.approx(1.0, abs=1e-10)


def test_complementary_mm_tapers_after_switch_on(default_cp: CutoffParameters) -> None:
    r_tail = default_cp.mm_switch_on + default_cp.mm_switch_width + 0.5
    s_ml, s_mm = default_cp.ml_mm_scales_complementary(
        r_tail, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF
    )
    assert s_ml == pytest.approx(0.0, abs=1e-12)
    assert s_mm == pytest.approx(0.0, abs=1e-12)


def test_pure_ml_region_has_no_mm(default_cp: CutoffParameters) -> None:
    r = default_cp.mm_switch_on - default_cp.ml_switch_width - 0.5
    assert default_cp.ml_scale(r, gamma_ml=GAMMA_ON) == pytest.approx(1.0)
    assert default_cp.mm_scale_complementary(
        r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF
    ) == pytest.approx(0.0, abs=1e-12)


def test_legacy_mm_scale_differs_from_complementary(default_cp: CutoffParameters) -> None:
    r = np.array([4.0, 5.5, 7.0])
    s_comp = default_cp.mm_scale_complementary(r, gamma_ml=GAMMA_ON, gamma_mm_off=GAMMA_OFF)
    s_legacy = default_cp.mm_scale(r, gamma_on=GAMMA_ON, gamma_off=GAMMA_OFF)
    assert not np.allclose(s_comp, s_legacy)


def test_simple_cosine_switches_match_endpoints() -> None:
    ml_w, mm_on, mm_w = 2.0, 7.0, 5.0
    assert ml_switch_simple(2.0, ml_w, mm_on) == pytest.approx(1.0)
    assert ml_switch_simple(8.0, ml_w, mm_on) == pytest.approx(0.0)
    assert mm_switch_simple(6.0, mm_on, mm_w) == pytest.approx(0.0)
    assert mm_switch_simple(9.5, mm_on, mm_w) == pytest.approx(0.5, abs=0.01)
    assert mm_switch_simple(12.0, mm_on, mm_w) == pytest.approx(1.0)


def test_gamma_constants_match_calculator_module() -> None:
    assert GAMMA_ON == CALC_GAMMA_ON
