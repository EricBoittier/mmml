"""PCA records its fit set and must not see held-out trajectories."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.acquisition.pca import (
    PCAFit,
    assert_no_leakage,
    fit_pca,
    project_jacobian_rows,
    transform_pca,
)


def test_pca_records_centering_scaling_and_retained_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    ids = [f"c{i}" for i in range(30)]
    fit = fit_pca(X, n_components=3, center=True, scale=True, fit_ids=ids)
    assert fit.n_components == 3
    assert fit.centered and fit.scaled
    assert fit.mean.shape == (6,)
    assert fit.scale.shape == (6,)
    assert 0.0 < fit.retained_variance <= 1.0 + 1e-9
    assert fit.fit_ids == tuple(ids)
    Z = transform_pca(X, fit)
    assert Z.shape == (30, 3)


def test_variance_threshold_keeps_enough_components():
    rng = np.random.default_rng(1)
    # Rank-2 signal plus tiny noise.
    base = rng.normal(size=(80, 2))
    X = np.concatenate([base, 1e-8 * rng.normal(size=(80, 5))], axis=1)
    fit = fit_pca(
        X,
        variance_threshold=0.99,
        scale=False,
        fit_ids=[f"i{i}" for i in range(80)],
    )
    assert fit.retained_variance >= 0.99 - 1e-9
    assert fit.n_components < X.shape[1]


def test_held_out_ids_are_rejected():
    X = np.eye(6)
    fit = fit_pca(X, n_components=2, fit_ids=["a", "b", "c", "d", "e", "f"])
    assert_no_leakage(fit, ["held-1"])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(fit, ["c"])


def test_fit_without_ids_cannot_prove_isolation_when_held_out_exist():
    fit = fit_pca(np.eye(4), n_components=2, fit_ids=[])
    with pytest.raises(ValueError, match="no fit_ids"):
        assert_no_leakage(fit, ["x"])


def test_jacobian_projection_does_not_center_rows():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(10, 5)) + 3.0
    fit = fit_pca(
        J, n_components=2, center=True, scale=False, fit_ids=["j"], kind="jacobian_basis"
    )
    centered = transform_pca(J, fit, apply_center=True)
    uncentered = project_jacobian_rows(J, fit)
    assert not np.allclose(centered, uncentered)
    # Uncentered projection is J @ components.T after scaling only.
    expected = J / fit.scale @ fit.components.T
    np.testing.assert_allclose(uncentered, expected)


def test_pca_roundtrip_dict():
    fit = fit_pca(np.random.default_rng(0).normal(size=(8, 5)), n_components=2, fit_ids=["u"])
    clone = PCAFit.from_dict(fit.to_dict())
    assert clone.n_components == fit.n_components
    np.testing.assert_allclose(clone.components, fit.components)
