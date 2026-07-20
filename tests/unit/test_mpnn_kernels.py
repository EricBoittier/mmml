"""Parity tests for PhysNet-family shared MPNN kernels."""

from __future__ import annotations

import functools

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.physnetjax.physnetjax.models.mpnn_kernels import (
    COULOMB_PAIR_FACTOR_EV_A,
    calc_electrostatics_switches,
    encode_geometry_and_basis,
    pair_displacements,
    pair_electrostatics_energy,
    radial_spherical_basis,
)
from mmml.models.physnetjax.physnetjax.models.model import PhysNet
from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet


def _reference_switches(
    displacements,
    batch_mask,
    *,
    switch_start,
    switch_end,
    electrostatics_off_start,
    electrostatics_off_end,
    electrostatics_damping_sigma,
):
    """Inline copy of the pre-extraction PhysNet/Spooky switch kernel."""
    eps = 1e-6
    min_dist = 0.01
    displacements = jnp.nan_to_num(displacements, nan=0.0, posinf=0.0, neginf=0.0)
    displacements = displacements + (1 - batch_mask)[..., None]
    squared_distances = jnp.sum(displacements**2, axis=1)
    distances = jnp.sqrt(jnp.maximum(squared_distances, min_dist**2))
    switch_dist = e3x.nn.smooth_switch(distances, switch_start, switch_end)
    off_dist = 1.0 - e3x.nn.smooth_switch(
        distances, electrostatics_off_start, electrostatics_off_end
    )
    switch_dist = jnp.clip(switch_dist, 0.0, 1.0)
    off_dist = jnp.clip(off_dist, 0.0, 1.0)
    one_minus_switch_dist = 1 - switch_dist
    safe_distances = distances + eps
    r1 = switch_dist / jnp.sqrt(squared_distances + 1.0)
    r2 = one_minus_switch_dist / safe_distances
    r = r1 + r2
    if electrostatics_damping_sigma > 0.0:
        sigma = jnp.asarray(electrostatics_damping_sigma, dtype=distances.dtype)
        r *= jax.scipy.special.erf(distances / sigma)
    eshift = safe_distances / (switch_end**2) - 2.0 / switch_end
    off_dist *= batch_mask
    eshift *= batch_mask
    r = jnp.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    off_dist = jnp.nan_to_num(off_dist, nan=0.0, posinf=0.0, neginf=0.0)
    eshift = jnp.nan_to_num(eshift, nan=0.0, posinf=0.0, neginf=0.0)
    return r, off_dist, eshift


@pytest.fixture
def water_edges():
    # O–H1, O–H2, H1–H2 (and reverse) style pairs for a 3-atom molecule
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        dtype=jnp.float32,
    )
    dst = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    src = jnp.asarray([1, 2, 0, 2, 0, 1], dtype=jnp.int32)
    batch_mask = jnp.ones(dst.shape[0], dtype=jnp.float32)
    return positions, dst, src, batch_mask


def test_calc_electrostatics_switches_matches_reference(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    kwargs = dict(
        switch_start=1.0,
        switch_end=10.0,
        electrostatics_off_start=8.0,
        electrostatics_off_end=10.0,
        electrostatics_damping_sigma=4.0,
    )
    got = calc_electrostatics_switches(disp, batch_mask, **kwargs)
    ref = _reference_switches(disp, batch_mask, **kwargs)
    for a, b in zip(got, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)


def test_calc_electrostatics_switches_zero_damping(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    kwargs = dict(
        switch_start=1.0,
        switch_end=10.0,
        electrostatics_off_start=8.0,
        electrostatics_off_end=10.0,
        electrostatics_damping_sigma=0.0,
    )
    got = calc_electrostatics_switches(disp, batch_mask, **kwargs)
    ref = _reference_switches(disp, batch_mask, **kwargs)
    for a, b in zip(got, ref):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)


def test_radial_spherical_basis_matches_inline(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    basis, disp_out = radial_spherical_basis(
        disp,
        num_basis_functions=8,
        max_degree=1,
        cutoff=5.0,
        batch_mask=batch_mask,
    )
    np.testing.assert_array_equal(np.asarray(disp_out), np.asarray(disp))

    m = batch_mask.astype(disp.dtype).reshape(-1, 1)
    basis_disp = disp + (1.0 - m)
    ref = e3x.nn.basis(
        basis_disp,
        num=8,
        max_degree=1,
        radial_fn=e3x.nn.exponential_chebyshev,
        cutoff_fn=functools.partial(e3x.nn.smooth_cutoff, cutoff=5.0),
    )
    ref = ref * batch_mask.astype(ref.dtype).reshape(-1, *([1] * (ref.ndim - 1)))
    np.testing.assert_allclose(np.asarray(basis), np.asarray(ref), rtol=0.0, atol=0.0)


def test_pair_electrostatics_energy_finite(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    r, off, eshift = calc_electrostatics_switches(
        disp,
        batch_mask,
        switch_start=1.0,
        switch_end=10.0,
        electrostatics_off_start=8.0,
        electrostatics_off_end=10.0,
        electrostatics_damping_sigma=4.0,
    )
    q = jnp.asarray([0.0, 0.5, -0.5], dtype=jnp.float32)
    segments = jnp.zeros(3, dtype=jnp.int32)
    atomic, batch = pair_electrostatics_energy(
        q, r, off, eshift, dst, src, batch_mask, segments, 1
    )
    assert atomic.shape == (3, 1, 1, 1)
    assert batch.shape == (1, 1, 1, 1)
    assert np.all(np.isfinite(np.asarray(atomic)))
    assert float(COULOMB_PAIR_FACTOR_EV_A) == pytest.approx(7.199822675975274)


def test_physnet_and_spooky_delegate_switches(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    kwargs = dict(
        switch_start=1.0,
        switch_end=10.0,
        electrostatics_off_start=8.0,
        electrostatics_off_end=10.0,
        electrostatics_damping_sigma=4.0,
    )
    shared = calc_electrostatics_switches(disp, batch_mask, **kwargs)
    phys = PhysNet(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=8,
        cutoff=5.0,
        max_padded_atoms=3,
        charges=True,
        zbl=False,
        **{k: kwargs[k] for k in (
            "switch_start",
            "switch_end",
            "electrostatics_off_start",
            "electrostatics_off_end",
            "electrostatics_damping_sigma",
        )},
    )
    spooky = SpookyPhysNet(
        features=8,
        max_degree=1,
        num_iterations=1,
        num_basis_functions=8,
        cutoff=5.0,
        max_padded_atoms=3,
        charges=True,
        zbl=False,
        **{k: kwargs[k] for k in (
            "switch_start",
            "switch_end",
            "electrostatics_off_start",
            "electrostatics_off_end",
            "electrostatics_damping_sigma",
        )},
    )
    for model in (phys, spooky):
        got = model._calc_switches(disp, batch_mask)
        for a, b in zip(got, shared):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)


def test_encode_geometry_and_basis_reciprocal_bernstein(water_edges):
    positions, dst, src, batch_mask = water_edges
    basis, disp = encode_geometry_and_basis(
        positions,
        dst,
        src,
        num_basis_functions=8,
        max_degree=1,
        cutoff=5.0,
        radial_fn=e3x.nn.reciprocal_bernstein,
        batch_mask=batch_mask,
    )
    assert basis.shape[0] == disp.shape[0]
    assert np.all(np.isfinite(np.asarray(basis)))


def test_edge_mask_zeros_basis(water_edges):
    positions, dst, src, batch_mask = water_edges
    disp = pair_displacements(positions, dst, src)
    edge_mask = jnp.asarray([1, 1, 0, 0, 1, 1], dtype=jnp.float32)
    basis, _ = radial_spherical_basis(
        disp,
        num_basis_functions=4,
        max_degree=0,
        cutoff=5.0,
        batch_mask=batch_mask,
        edge_mask=edge_mask,
    )
    # Masked edges (indices 2,3) must be exactly zero
    assert np.allclose(np.asarray(basis[2]), 0.0)
    assert np.allclose(np.asarray(basis[3]), 0.0)
    assert np.any(np.asarray(basis[0]) != 0.0)
