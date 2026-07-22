"""Unit tests for the pure-numeric helpers in mmml.spectra.spectra_md.

Covers autocorrelation/spectrum math that has no CHARMM/GPU/ML-calculator
dependency; ASE-trajectory/HDF5 extraction paths are left to manual/
integration testing (see docs/scientific-code.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.spectra.spectra_md import (
    FS_INV_TO_CM_INV,
    _freq_mask,
    _make_window,
    _next_pow2,
    autocorrelation,
    compute_magnetic_dipoles,
    correlation_to_spectrum,
    cross_correlation,
    noda_2d,
    polarizability_autocorrelation,
    raman_to_spectrum,
    stft_vector,
)


def test_next_pow2():
    assert _next_pow2(1) == 1
    assert _next_pow2(5) == 8
    assert _next_pow2(8) == 8
    assert _next_pow2(9) == 16


@pytest.mark.parametrize("kind,expected_len", [("hann", 10), ("blackman", 10), ("gaussian", 10)])
def test_make_window_known_kinds(kind, expected_len):
    w = _make_window(expected_len, kind)
    assert w.shape == (expected_len,)
    assert np.all(np.isfinite(w))


def test_make_window_unknown_kind_is_boxcar():
    w = _make_window(6, "not-a-real-window")
    np.testing.assert_allclose(w, np.ones(6))


def test_autocorrelation_zero_lag_equals_mean_square():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(200, 3))
    acf = autocorrelation(signal)
    assert acf.shape == (200,)
    expected_zero_lag = np.sum(signal * signal) / 200.0
    assert acf[0] == pytest.approx(expected_zero_lag, rel=1e-6)


def test_autocorrelation_constant_signal_is_flat():
    signal = np.ones((50, 3))
    acf = autocorrelation(signal)
    # constant signal: <x(0).x(tau)> == 3 at every lag
    np.testing.assert_allclose(acf, np.full(50, 3.0), rtol=1e-6)


def test_cross_correlation_of_signal_with_itself_matches_autocorrelation():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(64, 3))
    acf = autocorrelation(signal)
    ccf = cross_correlation(signal, signal)
    np.testing.assert_allclose(ccf, acf, rtol=1e-6)


def test_polarizability_autocorrelation_shapes_and_isotropic_zero_lag():
    rng = np.random.default_rng(2)
    T = 40
    alpha = rng.normal(size=(T, 3, 3))
    alpha = 0.5 * (alpha + alpha.transpose(0, 2, 1))  # symmetric, like a real polarizability
    acf_iso, acf_aniso = polarizability_autocorrelation(alpha)
    assert acf_iso.shape == (T,)
    assert acf_aniso.shape == (T,)
    alpha_iso = np.trace(alpha, axis1=1, axis2=2) / 3.0
    assert acf_iso[0] == pytest.approx(np.mean(alpha_iso ** 2), rel=1e-6)


def test_correlation_to_spectrum_freq_axis_and_nonneg_harmonic():
    dt_fs = 0.5
    n = 128
    t = np.arange(n) * dt_fs
    # a damped cosine at a known frequency
    freq_fs_inv = 0.01
    corr = np.cos(2 * np.pi * freq_fs_inv * t) * np.exp(-t / 50.0)

    freq_cm, spec = correlation_to_spectrum(corr, dt_fs, window="hann", zero_pad=4, qcf="harmonic")

    assert freq_cm[0] == pytest.approx(0.0)
    # rfftfreq is monotonically increasing
    assert np.all(np.diff(freq_cm) > 0)
    # harmonic QCF forces non-negativity
    assert np.all(spec >= 0.0)
    # peak should sit near the expected wavenumber
    expected_cm = freq_fs_inv * FS_INV_TO_CM_INV
    peak_cm = freq_cm[np.argmax(spec)]
    assert peak_cm == pytest.approx(expected_cm, rel=0.1)


def test_correlation_to_spectrum_classical_qcf_scales_by_omega_squared():
    dt_fs = 1.0
    corr = np.zeros(64)
    corr[0] = 1.0
    freq_h, spec_h = correlation_to_spectrum(corr, dt_fs, window=None, zero_pad=2, qcf="harmonic")
    freq_c, spec_c = correlation_to_spectrum(corr, dt_fs, window=None, zero_pad=2, qcf="classical")
    np.testing.assert_allclose(freq_h, freq_c)
    # away from the qcf==0 clamp, classical/harmonic ratio should track omega
    mask = freq_h > 10.0
    ratio = spec_c[mask] / np.maximum(spec_h[mask], 1e-12)
    np.testing.assert_allclose(ratio, freq_h[mask], rtol=1e-3)


def test_raman_to_spectrum_returns_consistent_shapes_and_total():
    rng = np.random.default_rng(3)
    T = 50
    acf_iso = rng.normal(size=T)
    acf_aniso = rng.normal(size=T)
    freq_cm, par, perp, total = raman_to_spectrum(acf_iso, acf_aniso, dt_fs=1.0)
    assert freq_cm.shape == par.shape == perp.shape == total.shape
    np.testing.assert_allclose(total, par + perp)


def test_compute_magnetic_dipoles_shape_and_zero_charge():
    T, N = 5, 3
    rng = np.random.default_rng(4)
    positions = rng.normal(size=(T, N, 3))
    velocities = rng.normal(size=(T, N, 3))
    charges = np.zeros((T, N))
    m = compute_magnetic_dipoles(positions, velocities, charges)
    assert m.shape == (T, 3)
    np.testing.assert_allclose(m, 0.0)


def test_stft_vector_shapes():
    rng = np.random.default_rng(5)
    T = 100
    signal = rng.normal(size=(T, 3))
    t_centres, freq_cm, power, ft_complex = stft_vector(
        signal, dt_fs=1.0, window_frames=20, stride_frames=10, window_fn="hann", zero_pad=2
    )
    n_windows = len(t_centres)
    assert n_windows > 0
    assert power.shape == (n_windows, len(freq_cm))
    assert ft_complex.shape == (n_windows, len(freq_cm), 3)


def test_noda_2d_shapes_and_symmetric_sync():
    rng = np.random.default_rng(6)
    spectrogram = rng.normal(size=(30, 8))
    sync, asynch = noda_2d(spectrogram)
    assert sync.shape == (8, 8)
    assert asynch.shape == (8, 8)
    # synchronous map (a covariance) must be symmetric
    np.testing.assert_allclose(sync, sync.T, atol=1e-10)


def test_freq_mask_inclusive_bounds():
    freq = np.array([0.0, 5.0, 10.0, 15.0])
    mask = _freq_mask(freq, 5.0, 10.0)
    np.testing.assert_array_equal(mask, [False, True, True, False])
