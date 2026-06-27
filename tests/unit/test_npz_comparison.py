"""Unit tests for NPZ comparison analysis (issue #12)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.analysis.npz_comparison import (
    align_npz_arrays,
    compare_npz_arrays,
    compute_element_force_metrics,
    compute_force_metrics,
    compute_per_atom_force_metrics,
    compute_scalar_metrics,
    plot_comparison,
    write_comparison_report,
)
from mmml.interfaces.pyscf4gpuInterface.finite_difference import (
    central_difference_gradient,
)


def test_scalar_metrics_perfect():
    x = np.linspace(0.0, 1.0, 20)
    m = compute_scalar_metrics(x, x)
    assert m.mae == pytest.approx(0.0, abs=1e-12)
    assert m.r2 == pytest.approx(1.0, abs=1e-12)


def test_force_metrics_and_element_breakdown():
    rng = np.random.default_rng(0)
    n, nat = 8, 4
    z = np.array([6, 1, 1, 1])
    z_batch = np.tile(z, (n, 1))
    targ = rng.normal(size=(n, nat, 3))
    pred = targ + rng.normal(scale=0.05, size=(n, nat, 3))

    fm = compute_force_metrics(pred, targ)
    assert fm["n_components"] == n * nat * 3
    assert fm["mae"] > 0

    pa = compute_per_atom_force_metrics(pred, targ, z_batch)
    assert len(pa["mae_per_atom"]) == nat
    assert pa["elements_per_atom"] == ["C", "H", "H", "H"]

    pe = compute_element_force_metrics(pred, targ, z_batch)
    assert set(pe.keys()) == {"C", "H"}
    assert pe["C"]["n"] == n
    assert pe["H"]["n"] == 3 * n


def test_align_and_compare_npz(tmp_path: Path):
    n, nat = 5, 3
    z = np.array([8, 1, 1])
    e_ref = np.arange(n, dtype=np.float64)
    e_pred = e_ref + 0.01
    f_ref = np.ones((n, nat, 3))
    f_pred = f_ref + 0.02
    d_ref = np.zeros((n, 3))
    d_pred = d_ref + 0.001

    ref = {
        "E": e_ref,
        "F": f_ref,
        "D": d_ref,
        "Z": np.tile(z, (n, 1)),
    }
    pred = {"E": e_pred, "F": f_pred, "D": d_pred, "Z": np.tile(z, (n, 1))}

    aligned = align_npz_arrays(ref, pred, max_frames=3)
    assert aligned["E_ref"].shape == (3,)
    metrics = compare_npz_arrays(aligned)
    assert metrics["n_frames"] == 3
    assert metrics["energy"]["mae"] == pytest.approx(0.01)
    assert "per_element_forces" in metrics

    plots = plot_comparison(
        aligned, metrics, tmp_path, energy_unit="eV", force_unit="eV/Å"
    )
    assert plots
    for p in plots:
        assert p.exists()

    metrics_path = write_comparison_report(
        metrics, tmp_path, reference="ref.npz", predictions="pred.npz", plot_paths=plots
    )
    loaded = json.loads(metrics_path.read_text())
    assert loaded["energy"]["mae"] == pytest.approx(0.01)


def test_central_difference_quadratic():
    # E = 0.5 * k * |r|^2 in Hartree with k in Ha/Å^2 -> grad in Ha/Å
  # Use simple harmonic: E = sum (x^2 + y^2 + z^2) in Hartree
    def energy_fn(r):
        return float(np.sum(r**2))

    r0 = np.array([[0.5, 0.0, 0.0], [0.0, 0.3, 0.1]])
    grad = central_difference_gradient(energy_fn, r0, step_ang=1e-4)
    # dE/dx = 2x Ha/Å -> convert to Ha/Bohr
    expected_ha_ang = 2.0 * r0
    from mmml.data.units import EV_ANGSTROM_TO_HARTREE_BOHR, HARTREE_TO_EV

    expected = expected_ha_ang * HARTREE_TO_EV * EV_ANGSTROM_TO_HARTREE_BOHR
    np.testing.assert_allclose(grad, expected, rtol=1e-3, atol=1e-3)
