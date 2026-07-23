"""Native Ewald under NpT: live box_override + tolerance-gated k-grid rebuild."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mmml.interfaces.pycharmmInterface.ewald_native import (
    EWALD_NPT_KGRID_REBUILD_TOLERANCE_A,
    ewald_npt_kgrid_cache_bin,
)
from mmml.models.ewald_hybrid_coulomb import (
    ewald_static_params_from_box_length,
    hybrid_ewald_coulomb_energy,
    hybrid_ewald_coulomb_energy_with_cell,
)


def test_ewald_npt_kgrid_cache_bin_stable_within_tolerance():
    tol = EWALD_NPT_KGRID_REBUILD_TOLERANCE_A
    L0 = 30.0
    b0 = ewald_npt_kgrid_cache_bin(L0, tolerance_A=tol)
    assert ewald_npt_kgrid_cache_bin(L0 + 0.4 * tol, tolerance_A=tol) == b0
    assert ewald_npt_kgrid_cache_bin(L0 + 1.1 * tol, tolerance_A=tol) != b0


def test_hybrid_ewald_with_cell_matches_static_api():
    pos = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 1], dtype=jnp.int32)
    q = jnp.array([1.0, -1.0], dtype=jnp.float64)
    L = 40.0
    e_static = hybrid_ewald_coulomb_energy(
        pos, mid, q, box_length_A=L, real_space_cutoff_A=10.0
    )
    alpha, n_int = ewald_static_params_from_box_length(L, real_space_cutoff_A=10.0)
    e_cell = hybrid_ewald_coulomb_energy_with_cell(
        pos,
        mid,
        q,
        jnp.diag(jnp.array([L, L, L])),
        alpha=alpha,
        n_int=n_int,
        real_space_cutoff_A=10.0,
    )
    assert float(e_cell) == pytest.approx(float(e_static), rel=0, abs=1e-10)


def test_hybrid_ewald_with_cell_tracks_live_box_same_n_int():
    """Within a k-grid bin, energy must respond to L via traced cell (NpT)."""
    pos = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 1], dtype=jnp.int32)
    q = jnp.array([1.0, -1.0], dtype=jnp.float64)
    L0 = 30.0
    alpha, n_int = ewald_static_params_from_box_length(L0)
    e0 = float(
        hybrid_ewald_coulomb_energy_with_cell(
            pos, mid, q, jnp.diag(jnp.array([L0, L0, L0])), alpha=alpha, n_int=n_int
        )
    )
    L1 = L0 + 0.2  # well inside default 0.5 Å rebuild bin
    e1 = float(
        hybrid_ewald_coulomb_energy_with_cell(
            pos, mid, q, jnp.diag(jnp.array([L1, L1, L1])), alpha=alpha, n_int=n_int
        )
    )
    assert e0 != pytest.approx(e1, abs=1e-8)


def test_ewald_energy_with_cell_jittable_grad_through_box():
    """Forces w.r.t. positions remain finite when cell is a traced NpT input."""
    pos0 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    mid = jnp.array([0, 1], dtype=jnp.int32)
    chg = jnp.array([1.0, -1.0], dtype=jnp.float64)
    L = 40.0
    alpha, n_int = ewald_static_params_from_box_length(L, real_space_cutoff_A=10.0)
    cell = jnp.diag(jnp.array([L, L, L], dtype=jnp.float64))

    def e_fn(p):
        return hybrid_ewald_coulomb_energy_with_cell(
            p,
            mid,
            chg,
            cell,
            alpha=alpha,
            n_int=n_int,
            real_space_cutoff_A=10.0,
        )

    e, g = jax.jit(jax.value_and_grad(e_fn))(pos0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert np.isfinite(float(e))
