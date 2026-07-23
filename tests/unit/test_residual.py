"""Unit tests for mmml.residual: pure-JAX explicit electrostatics + residual NPZ builder."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from mmml.residual.electrostatics_jax import (
    KE_KCAL_ANG,
    _switch,
    build_pairs,
    elec_energy,
    elec_energy_and_forces,
)
from mmml.residual.build_residual import build_residual_npz


def test_build_pairs_all_i_lt_j_no_exclusions():
    pi, pj = build_pairs(4)
    pairs = set(zip(np.asarray(pi).tolist(), np.asarray(pj).tolist()))
    assert pairs == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_build_pairs_respects_exclusions_either_order():
    pi, pj = build_pairs(4, exclusions=[(1, 0), (2, 3)])
    pairs = set(zip(np.asarray(pi).tolist(), np.asarray(pj).tolist()))
    assert (0, 1) not in pairs
    assert (2, 3) not in pairs
    assert (0, 2) in pairs


def test_switch_is_one_below_ron_and_zero_above_roff():
    r = jnp.asarray([0.5, 5.0, 15.0])
    s = _switch(r, r_on=1.0, r_off=10.0)
    assert float(s[0]) == pytest.approx(1.0)
    assert float(s[2]) == pytest.approx(0.0)
    assert 0.0 < float(s[1]) < 1.0


def test_elec_energy_two_charges_matches_coulomb_law():
    R = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    q = jnp.asarray([1.0, -1.0])
    pi, pj = build_pairs(2)
    e = elec_energy(R, q, pi, pj)
    assert float(e) == pytest.approx(-KE_KCAL_ANG / 2.0, rel=1e-6)


def test_elec_energy_cutoff_switch_kills_energy_beyond_roff():
    R = jnp.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    q = jnp.asarray([1.0, 1.0])
    pi, pj = build_pairs(2)
    e = elec_energy(R, q, pi, pj, r_on=5.0, r_off=10.0)
    assert float(e) == pytest.approx(0.0, abs=1e-8)


def test_elec_energy_and_forces_matches_finite_difference():
    key = jax.random.PRNGKey(0)
    R = jax.random.normal(key, (6, 3)) * 3.0
    q = jnp.asarray(np.random.default_rng(1).uniform(-0.8, 0.8, 6))
    q = q - jnp.mean(q)
    pi, pj = build_pairs(6, exclusions=[(0, 1)])

    e, f = elec_energy_and_forces(R, q, pi, pj)

    h = 1e-4
    g = np.zeros((6, 3))
    for a in range(6):
        for d in range(3):
            Rp = R.at[a, d].add(h)
            Rm = R.at[a, d].add(-h)
            g[a, d] = -(
                float(elec_energy(Rp, q, pi, pj)) - float(elec_energy(Rm, q, pi, pj))
            ) / (2 * h)

    assert np.max(np.abs(np.asarray(f) - g)) < 1e-4
    assert np.isfinite(float(e))


def test_build_residual_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    M, N = 3, 5
    R = rng.normal(size=(M, N, 3)) * 2.0
    Z = rng.integers(1, 9, size=N)
    q = rng.uniform(-0.5, 0.5, N)
    q -= q.mean()
    E_ref = rng.normal(size=M)
    F_ref = rng.normal(size=(M, N, 3))
    pi, pj = build_pairs(N, exclusions=[(0, 1)])

    def elec_fn(Rf, q, pi, pj):
        e, f = elec_energy_and_forces(jnp.asarray(Rf), jnp.asarray(q), pi, pj, KE_KCAL_ANG)
        return float(e), np.asarray(f)

    out_path = tmp_path / "residual.npz"
    build_residual_npz(str(out_path), R, Z, E_ref, F_ref, q, pi, pj, elec_fn)

    assert out_path.exists()
    data = np.load(out_path)
    np.testing.assert_allclose(data["R"], R)
    np.testing.assert_allclose(data["Z"], Z)
    np.testing.assert_allclose(data["E_ref"], E_ref)
    np.testing.assert_allclose(data["F_ref"], F_ref)
    # E_residual = E_ref - E_expl_elec by construction
    np.testing.assert_allclose(data["E_residual"], data["E_ref"] - data["E_expl_elec"])
    np.testing.assert_allclose(data["F_residual"], data["F_ref"] - data["F_expl_elec"])
    assert data["E_expl_elec"].shape == (M,)
    assert data["F_expl_elec"].shape == (M, N, 3)


def test_build_residual_npz_zero_electrostatics_leaves_residual_equal_to_ref(tmp_path):
    rng = np.random.default_rng(2)
    M, N = 2, 4
    R = rng.normal(size=(M, N, 3))
    Z = rng.integers(1, 9, size=N)
    q = np.zeros(N)
    E_ref = rng.normal(size=M)
    F_ref = rng.normal(size=(M, N, 3))
    pi, pj = build_pairs(N)

    def zero_elec_fn(Rf, q, pi, pj):
        return 0.0, np.zeros_like(Rf)

    out_path = tmp_path / "zero_elec.npz"
    build_residual_npz(str(out_path), R, Z, E_ref, F_ref, q, pi, pj, zero_elec_fn)

    data = np.load(out_path)
    np.testing.assert_allclose(data["E_residual"], E_ref)
    np.testing.assert_allclose(data["F_residual"], F_ref)
