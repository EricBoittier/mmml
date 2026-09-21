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


@pytest.mark.parametrize("close_pair", ["first", "last"])
def test_sparse_dimer_padded_slots_add_no_forces(close_pair: str) -> None:
    """Sparse and dense dimer paths give the same forces when the cap has spare slots.

    Unused sparse slots carry index ``n_dimers``; a raw gather clamps it to the
    last pair, so a close last pair used to get its switched force added once
    per spare slot.
    """
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono, n_monomers, box = 5, 6, 40.0
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.7, size=(n_mono, 3))
    far = [[2, 2, 2], [20, 2, 2], [2, 20, 2], [2, 2, 20]]
    if close_pair == "last":  # pair (4, 5) about 4 Å apart, all others > 6 Å
        centers = far + [[20, 20, 20], [24, 20, 20]]
    else:  # pair (0, 1) about 3.5 Å apart
        centers = [[2, 2, 2], [5.5, 2, 2], [2, 20, 2], [2, 2, 20], [20, 20, 20], [20, 20, 2]]
    r0 = jnp.asarray(
        np.concatenate([np.asarray(c, float) + base + rng.normal(scale=0.05, size=base.shape) for c in centers])
    )
    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool))

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    cp = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)

    def evaluate(sparse: bool):
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
                ml_max_active_dimers=6 if sparse else None,
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

    dense, sparse = evaluate(False), evaluate(True)
    np.testing.assert_allclose(float(sparse.energy), float(dense.energy), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sparse.forces), np.asarray(dense.forces), atol=1e-3)
