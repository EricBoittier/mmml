"""Reweighting estimator and LJ theta mapping for liquid-observable fitting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.fit.lj_theta import LjTypeMap, init_theta, per_atom_lj, project_theta
from mmml.fit.reweight import (
    KB_KCAL_MOL_K,
    effective_sample_fraction,
    reweight_weights,
    reweighted_mean,
)

jax.config.update("jax_enable_x64", True)


def test_reweighted_harmonic_mean_and_gradient_match_analytic():
    # U_k(x) = k x^2 / 2 sampled at k0: <x^2>_k = kT / k, d<x^2>/dk = -kT / k^2.
    T, k0, k = 300.0, 2.0, 2.3
    kT = KB_KCAL_MOL_K * T
    x = np.random.default_rng(0).normal(0.0, np.sqrt(kT / k0), size=400_000)
    u_ref = 0.5 * k0 * x**2

    def mean_x2(kk):
        w = reweight_weights(0.5 * kk * x**2, u_ref, T)
        return reweighted_mean(jnp.asarray(x**2), w)

    assert float(mean_x2(k)) == pytest.approx(kT / k, rel=1e-2)
    assert float(jax.grad(mean_x2)(k)) == pytest.approx(-kT / k**2, rel=3e-2)


def test_effective_sample_fraction_limits():
    assert float(effective_sample_fraction(jnp.full(10, 0.1))) == pytest.approx(1.0)
    one_hot = jnp.zeros(10).at[3].set(1.0)
    assert float(effective_sample_fraction(one_hot)) == pytest.approx(0.1)


def test_lj_theta_scales_only_fitted_types_within_bounds():
    tmap = LjTypeMap.from_atc([0, 1, 2, 1], ["CG331", "HGA3", "OG2D3"], fit_types=["HGA3"])
    assert tmap.atom_type.tolist() == [-1, 0, -1, 0]
    theta = init_theta(tmap)
    theta = project_theta({"log_eps": theta["log_eps"] + 5.0, "log_sig": theta["log_sig"] - 5.0})
    rm, ep = per_atom_lj(theta, tmap, jnp.ones(4), -jnp.ones(4))
    np.testing.assert_allclose(np.asarray(rm), [1.0, 0.95, 1.0, 0.95])
    np.testing.assert_allclose(np.asarray(ep), [-1.0, -4.0, -1.0, -4.0])
