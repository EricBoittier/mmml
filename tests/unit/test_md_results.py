"""Tests for mmml.md.results.energy_drift_metrics."""

from __future__ import annotations

import numpy as np

from mmml.md.results import energy_drift_metrics


def test_flat_trace_has_zero_trend_and_zero_fluctuation():
    energies = np.full(50, -100.0)
    metrics = energy_drift_metrics(energies)
    assert metrics["energy_fluctuation_std_ev"] == 0.0
    assert abs(metrics["energy_trend_ev_per_frame"]) < 1e-9
    assert abs(metrics["energy_trend_total_ev"]) < 1e-9


def test_pure_linear_trend_recovers_the_exact_slope():
    """fluctuation_std is std of the raw trace (not detrended), so a ramp has
    nonzero std by construction; what a *pure* ramp guarantees is that the
    linear fit recovers the slope exactly (no residual noise to bias it)."""
    slope = 0.05
    frames = np.arange(100, dtype=float)
    energies = -500.0 + slope * frames
    metrics = energy_drift_metrics(energies)
    assert abs(metrics["energy_trend_ev_per_frame"] - slope) < 1e-9
    assert abs(metrics["energy_trend_total_ev"] - slope * 99) < 1e-9


def test_endpoint_delta_can_mislead_but_trend_reflects_the_whole_trace():
    """A trace that fluctuates but has near-zero net trend should report a
    small trend even if two arbitrary endpoints happen to differ a lot."""
    frames = np.arange(20, dtype=float)
    energies = -50.0 + 2.0 * np.sin(frames)  # oscillates, no systematic drift
    metrics = energy_drift_metrics(energies)
    assert metrics["energy_fluctuation_std_ev"] > 0.5
    # A pure oscillation over a couple of periods has ~zero linear trend,
    # much smaller than the fluctuation itself.
    assert abs(metrics["energy_trend_ev_per_frame"]) < 0.1


def test_single_frame_returns_zeros_for_trend():
    metrics = energy_drift_metrics(np.array([-42.0]))
    assert metrics["energy_trend_ev_per_frame"] == 0.0
    assert metrics["energy_trend_total_ev"] == 0.0
