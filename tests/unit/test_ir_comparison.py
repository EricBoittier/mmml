"""Unit tests for MCP IR comparison plotting."""

from __future__ import annotations

import numpy as np

from mmml.mcp.ir_comparison import (
    NIST_DCM_IR,
    build_post_smooth_variants,
    compute_all_method_spectra,
    normalize01,
    sticks_to_spectrum,
)


def test_normalize01_clips_to_unit_interval() -> None:
    y = np.array([0.0, 2.0, 4.0, -1.0])
    out = normalize01(y)
    assert out.min() >= 0.0
    assert out.max() == 1.0
    assert out[1] == 0.5


def test_sticks_to_spectrum_peak_near_frequency() -> None:
    grid = np.linspace(600.0, 3200.0, 2000)
    spec = sticks_to_spectrum(np.array([750.0]), np.array([1.0]), grid, fwhm=10.0)
    assert normalize01(spec).max() == 1.0
    assert normalize01(spec).argmax() == np.argmin(np.abs(grid - 750.0))


def test_compute_all_method_spectra_normalized() -> None:
    rng = np.random.default_rng(0)
    t = 512
    mm = rng.normal(size=(t, 3))
    ml = rng.normal(size=(t, 3))
    specs = compute_all_method_spectra(mm, ml, frame_dt_fs=1.0, n_grid=200)
    grid = specs.pop("_grid")["freq_cm"]
    assert len(grid) == 200
    for key, payload in specs.items():
        assert payload["mm"].max() <= 1.0 + 1e-12
        assert payload["ml"].max() <= 1.0 + 1e-12
        assert payload["mm"].min() >= 0.0
        assert payload["ml"].min() >= 0.0
    assert "06_gaussian_smooth" in specs
    assert len(NIST_DCM_IR) >= 6


def test_post_smooth_variants_bounded() -> None:
    grid = np.linspace(600.0, 3200.0, 500)
    mm = np.exp(-0.5 * ((grid - 750.0) / 40.0) ** 2)
    ml = mm * 0.8
    variants = build_post_smooth_variants(grid, mm, ml)
    assert len(variants) == 3
    for _key, _title, _desc, mm_s, ml_s in variants:
        assert mm_s.max() <= 1.0
        assert ml_s.max() <= 1.0
