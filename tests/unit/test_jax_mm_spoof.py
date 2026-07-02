"""Unit tests for JAX MM clone spoof ML potential."""

from __future__ import annotations

import os
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

# Spoof hybrid eval compiles on CPU; avoid GPU OOM in agent/CI sessions.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
    build_jax_mm_spoof_batch_apply,
    build_minimal_chain_bonded_evaluator,
    jax_mm_spoof_enabled,
    minimal_chain_bonded_system,
)


def test_jax_mm_spoof_enabled_from_args() -> None:
    from argparse import Namespace

    assert not jax_mm_spoof_enabled(None)
    assert jax_mm_spoof_enabled(Namespace(jax_mm_spoof=True))
    assert jax_mm_spoof_enabled(Namespace(ml_potential_mode="jax_mm_clone"))


def test_minimal_chain_bonded_energy_finite() -> None:
    topology, bonded = minimal_chain_bonded_system(5)
    assert topology.bonds.shape[0] == 4
    eval_fn = build_minimal_chain_bonded_evaluator(5, energy_unit="eV")
    pos = jnp.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.2, 0.0], [4.2, 0.5, 0.1], [5.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    energy, forces = eval_fn(pos)
    assert float(energy) > 0.0
    assert forces.shape == (5, 3)
    assert bool(jnp.all(jnp.isfinite(forces)))


def test_jax_mm_spoof_batch_apply_monomer_and_dimer() -> None:
    mono_eval = build_minimal_chain_bonded_evaluator(5, energy_unit="eV")
    apply = build_jax_mm_spoof_batch_apply(
        atoms_per_monomer=5,
        max_atoms=10,
        monomer_eval=mono_eval,
    )
    batch_n = jnp.array([5, 10], dtype=jnp.int32)
    R = jnp.zeros((2, 10, 3), dtype=jnp.float64)
    for i in range(5):
        R = R.at[0, i].set(jnp.array([float(i), 0.1 * i, 0.0]))
        R = R.at[1, i].set(jnp.array([float(i), 0.0, 0.0]))
        R = R.at[1, 5 + i].set(jnp.array([float(i) + 6.0, 0.5, 0.0]))
    Z = jnp.ones((2, 10), dtype=jnp.int32)
    out = apply(Z.reshape(-1), R.reshape(-1, 3), batch_n)
    assert out["energy"].shape == (2,)
    assert bool(jnp.all(jnp.isfinite(out["energy"])))
    assert out["forces"].shape == (20, 3)
    assert bool(jnp.all(jnp.isfinite(out["forces"])))


def test_setup_calculator_jax_mm_spoof_hybrid_eval() -> None:
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    n_mono = 5
    n_monomers = 4
    n_atoms = n_mono * n_monomers
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    r0 = np.random.default_rng(0).normal(size=(n_atoms, 3))
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
            ml_sparse_dimers=False,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )
        out = spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CutoffParameters(),
            doML=True,
            doMM=False,
            doML_dimer=True,
            box=jnp.asarray([[box, 0, 0], [0, box, 0], [0, 0, box]], dtype=jnp.float64),
        )
    assert bool(jnp.isfinite(out.energy))
    assert out.forces.shape == (n_atoms, 3)
    assert bool(jnp.all(jnp.isfinite(out.forces)))
