"""Unit tests for MCP IR comparison plotting."""

from __future__ import annotations

import numpy as np

from mmml.mcp.ir_comparison import (
    NIST_DCM_IR,
    build_comparison_spectra,
    sticks_to_spectrum,
)


def test_sticks_to_spectrum_peak_near_frequency() -> None:
    grid = np.linspace(600.0, 3200.0, 2000)
    spec = sticks_to_spectrum(np.array([750.0]), np.array([1.0]), grid, fwhm=10.0)
    assert spec.argmax() == np.argmin(np.abs(grid - 750.0))


def test_build_comparison_spectra_keys() -> None:
    freq = np.linspace(600.0, 3200.0, 500)
    broad = np.exp(-0.5 * ((freq - 750.0) / 40.0) ** 2)
    spectra = build_comparison_spectra(
        mm_freq=freq,
        mm_ir=broad,
        ml_freq=freq,
        ml_ir=broad * 0.8,
        harmonic_freqs=np.array([750.0, 3000.0]),
        harmonic_int=np.array([1.0, 0.5]),
    )
    assert set(spectra) == {"mm", "ml", "nist", "harmonic"}
    assert spectra["mm"].intensity.max() == 1.0
    assert len(NIST_DCM_IR) >= 6
