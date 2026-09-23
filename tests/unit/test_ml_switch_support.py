"""ML switch support: s and ds vanish at both handoff endpoints, including PBC."""

from __future__ import annotations

import os
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.calculators.ase_fragment_hybrid import numpy_ml_switch_scale_and_deriv
from mmml.interfaces.pycharmmInterface.calculator_utils import ml_switch_scale
from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    sparse_dimer_active_radius,
)


MM_ON = 6.0
WIDTH = 1.5
INNER = MM_ON - WIDTH


def test_switch_value_and_deriv_vanish_at_both_endpoints():
    """s=1, ds=0 at the inner edge; s=0, ds=0 at mm_switch_on and beyond."""
    r = jnp.asarray(
        [INNER - 0.2, INNER, INNER + 0.2, MM_ON - 0.2, MM_ON, MM_ON + 0.2]
    )
    s = ml_switch_scale(r, mm_switch_on=MM_ON, ml_switch_width=WIDTH)
    ds = jax.vmap(
        jax.grad(lambda x: ml_switch_scale(x, mm_switch_on=MM_ON, ml_switch_width=WIDTH))
    )(r)
    np.testing.assert_allclose(s[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(s[1], 1.0, atol=1e-12)
    assert float(s[2]) < 1.0
    assert float(s[3]) > 0.0
    np.testing.assert_allclose(s[4], 0.0, atol=1e-12)
    np.testing.assert_allclose(s[5], 0.0, atol=1e-12)
    np.testing.assert_allclose(ds[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(ds[4], 0.0, atol=1e-10)
    np.testing.assert_allclose(ds[5], 0.0, atol=1e-10)
    for ri, si, dsi in zip(np.asarray(r), np.asarray(s), np.asarray(ds)):
        n_s, n_ds = numpy_ml_switch_scale_and_deriv(
            float(ri), mm_switch_on=MM_ON, ml_switch_width=WIDTH
        )
        assert n_s == pytest.approx(float(si), abs=1e-12)
        assert n_ds == pytest.approx(float(dsi), abs=1e-8)


def test_active_radius_matches_outer_endpoint():
    assert sparse_dimer_active_radius(MM_ON, WIDTH) == MM_ON


def _two_monomer_coords(sep: float, *, box: float, wrap: bool, n_mono: int = 5):
    rng = np.random.default_rng(2)
    base = rng.normal(scale=0.15, size=(n_mono, 3))
    c0 = np.array([2.0, 2.0, 2.0])
    c1 = c0 + np.array([sep, 0.0, 0.0])
    if wrap:
        c1[0] = c1[0] - box  # B is across the periodic face; MIC sep is still ``sep``
    r0 = np.concatenate([c0 + base, c1 + base])
    return r0


def _eval_two_monomer(sep: float, *, sparse: bool, wrap: bool, box: float = 20.0):
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, n_monomers = 5, 2
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    r0 = jnp.asarray(_two_monomer_coords(sep, box=box, wrap=wrap, n_mono=n_mono))
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.ones((1,), dtype=bool),
    )

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=MM_ON, ml_switch_width=WIDTH)
    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=fake_build_mm,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=n_mono,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=False,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=10,
            cell=box,
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=sparse,
            ml_max_active_dimers=1 if sparse else None,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=r0,
            n_monomers=n_monomers,
            cutoff_params=cp,
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        return spherical_fn(
            atomic_numbers=z,
            positions=r0,
            n_monomers=n_monomers,
            cutoff_params=cp,
            doML=True,
            doMM=False,
            doML_dimer=True,
            box=jnp.eye(3) * box,
        )


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize(
    "sep",
    [INNER - 0.3, INNER + 0.3, MM_ON - 0.3, MM_ON + 0.3],
)
def test_sparse_matches_dense_across_switch_boundaries(sep: float, wrap: bool) -> None:
    """Energies and forces agree on both sides of each handoff edge, with PBC."""
    dense = _eval_two_monomer(sep, sparse=False, wrap=wrap)
    sparse = _eval_two_monomer(sep, sparse=True, wrap=wrap)
    np.testing.assert_allclose(float(sparse.energy), float(dense.energy), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(sparse.forces), np.asarray(dense.forces), atol=1e-3)
