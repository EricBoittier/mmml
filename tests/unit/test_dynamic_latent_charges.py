"""Unit tests for the Mode E (latent_dynamic) weighted scatter-average core."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.dynamic_latent_charges import weighted_scatter_average


def test_single_slot_full_weight_is_a_plain_scatter():
    values = jnp.array([[0.3, -0.3, 0.0]])
    global_idx = jnp.array([[0, 1, 0]])  # atom 2 unused (mask False)
    weights = jnp.array([1.0])
    mask = jnp.array([[True, True, False]])
    out = weighted_scatter_average(values, global_idx, weights, mask, n_atoms=3)
    assert np.allclose(np.asarray(out), [0.3, -0.3, 0.0], atol=1e-12)


def test_two_partners_averaged_by_weight():
    # Monomer atom 0 gets two independent estimates (from two active dimer
    # partners), 0.2 (weight 1.0, close partner) and 0.6 (weight 0.5, farther
    # partner): weighted mean = (0.2*1.0 + 0.6*0.5) / (1.0 + 0.5).
    values = jnp.array([[0.2], [0.6]])
    global_idx = jnp.array([[0], [0]])
    weights = jnp.array([1.0, 0.5])
    mask = jnp.array([[True], [True]])
    out = weighted_scatter_average(values, global_idx, weights, mask, n_atoms=1)
    expected = (0.2 * 1.0 + 0.6 * 0.5) / 1.5
    assert float(out[0]) == pytest.approx(expected, abs=1e-12)


def test_zero_weight_atom_returns_zero_not_nan():
    # No active partner references this atom -> weight_sum == 0, must not NaN.
    values = jnp.array([[0.5]])
    global_idx = jnp.array([[0]])
    weights = jnp.array([0.0])
    mask = jnp.array([[True]])
    out = weighted_scatter_average(values, global_idx, weights, mask, n_atoms=2)
    assert np.all(np.isfinite(np.asarray(out)))
    assert np.allclose(np.asarray(out), [0.0, 0.0])


def test_masked_slots_do_not_contribute():
    values = jnp.array([[10.0, 20.0]])
    global_idx = jnp.array([[0, 1]])
    weights = jnp.array([1.0])
    mask = jnp.array([[True, False]])  # atom 1's slot is padding
    out = weighted_scatter_average(values, global_idx, weights, mask, n_atoms=2)
    assert float(out[0]) == pytest.approx(10.0, abs=1e-12)
    assert float(out[1]) == pytest.approx(0.0, abs=1e-12)
