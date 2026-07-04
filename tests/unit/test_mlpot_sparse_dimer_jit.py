"""JIT safety for sparse ML dimer COM distances under PBC."""

from __future__ import annotations

import os
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters


def test_sparse_dimer_jit_with_traced_box() -> None:
    """Sparse dimer filtering must not Python-branch on traced cell values."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono = 5
    n_monomers = 8
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    rng = np.random.default_rng(0)
    r0 = rng.normal(size=(n_atoms, 3))
    box = 27.0
    fake_mm_fn = lambda *args, **kwargs: (
        jnp.array(0.0, dtype=jnp.float32),
        jnp.zeros((n_atoms, 3), dtype=jnp.float32),
    )
    fake_update_fn = lambda *args, **kwargs: (
        jnp.zeros((1, 2), dtype=jnp.int32),
        jnp.ones((1,), dtype=bool),
    )

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

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
            ml_sparse_dimers=True,
            ml_max_active_dimers=10,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(mm_switch_on=12.0),
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        box_mat = jnp.asarray(
            [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]],
            dtype=jnp.float64,
        )
        out = spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(mm_switch_on=12.0),
            doML=True,
            doMM=False,
            doML_dimer=True,
            box=box_mat,
        )
    assert bool(jnp.isfinite(out.energy))
    assert out.forces.shape == (n_atoms, 3)
    assert bool(jnp.all(jnp.isfinite(out.forces)))
