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


def test_jax_mm_spoof_batch_apply_heterogeneous_sizes() -> None:
    """3+6 dimers must not be confused with a 6-atom monomer (N alone is ambiguous)."""
    from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
        resolve_monomer_bonded_evaluators,
    )

    per = [3, 6, 3]
    evals = resolve_monomer_bonded_evaluators(per)
    apply = build_jax_mm_spoof_batch_apply(
        atoms_per_monomer=per,
        max_atoms=12,
        monomer_evals=evals,
    )
    # batch: mono(3), mono(6), dimer(3+6)=9
    batch_n = jnp.array([3, 6, 9], dtype=jnp.int32)
    batch_n_a = jnp.array([3, 6, 3], dtype=jnp.int32)
    R = jnp.zeros((3, 12, 3), dtype=jnp.float64)
    for i in range(3):
        R = R.at[0, i].set(jnp.array([float(i), 0.0, 0.0]))
        R = R.at[2, i].set(jnp.array([float(i), 0.0, 0.0]))
    for i in range(6):
        R = R.at[1, i].set(jnp.array([float(i), 0.2, 0.0]))
        R = R.at[2, 3 + i].set(jnp.array([float(i) + 4.0, 1.0, 0.0]))
    Z = jnp.ones((3, 12), dtype=jnp.int32)
    out = apply(Z.reshape(-1), R.reshape(-1, 3), batch_n, batch_n_a)
    assert out["energy"].shape == (3,)
    assert bool(jnp.all(jnp.isfinite(out["energy"])))
    assert out["forces"].shape == (36, 3)
    assert bool(jnp.all(jnp.isfinite(out["forces"])))
    # Heterogeneous dimer should be more than either monomer alone when far apart
    # (soft repulsion is small, bonded energy dominates); at least all finite.
    assert float(out["energy"][2]) != 0.0


def test_setup_calculator_jax_mm_spoof_heterogeneous() -> None:
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    per = [3, 6, 3, 6]
    n_monomers = len(per)
    n_atoms = sum(per)
    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    r0 = np.random.default_rng(1).normal(size=(n_atoms, 3)) * 0.5
    # Separate monomers along x so soft repulsion stays finite
    off = 0
    for i, n in enumerate(per):
        r0[off : off + n, 0] += float(i) * 6.0
        off += n
    box = 40.0
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
            ATOMS_PER_MONOMER=per,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=False,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=12,
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


def test_soft_repulsion_finite_near_contact() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
        _inter_monomer_soft_repulsion,
    )

    R = jnp.zeros((10, 3), dtype=jnp.float32)
    R = R.at[0].set(jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32))
    # Near-overlap A–B pair formerly overflowed float32 via an N×N 1/r^12 matrix.
    R = R.at[5].set(jnp.array([1e-4, 0.0, 0.0], dtype=jnp.float32))
    for i in range(1, 5):
        R = R.at[i].set(jnp.array([float(i) * 1.5, 0.0, 0.0], dtype=jnp.float32))
        R = R.at[5 + i].set(jnp.array([float(i) * 1.5 + 4.0, 0.5, 0.0], dtype=jnp.float32))
    e, f = _inter_monomer_soft_repulsion(R, 5, 5)
    assert bool(jnp.isfinite(e))
    assert bool(jnp.all(jnp.isfinite(f)))
    assert float(jnp.max(jnp.abs(f))) < 1.0e6


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


def test_spoof_psf_monomer_matches_cgenff_bonded_components() -> None:
    """``jax_mm_spoof`` PSF slice must match full-system CGenFF bonded for that monomer."""
    from pathlib import Path

    from jax_md.mm_forcefields.io.charmm import parse_pdb_simple

    from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        filter_bonded_topology_for_mm,
        load_cgenff_bonded_from_psf,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.jax_mm_spoof import (
        load_monomer_bonded_components_from_psf,
        load_monomer_bonded_evaluator_from_psf,
    )

    aco_psf = Path("tests/functionality/pycharmmETC/psf/aco-1.psf")
    aco_pdb = Path("tests/functionality/pycharmmETC/pdb/aco.pdb")
    if not aco_psf.is_file() or not aco_pdb.is_file():
        pytest.skip("ACO PSF/PDB fixtures missing")

    _, positions = parse_pdb_simple(str(aco_pdb))
    pos = jnp.asarray(positions, dtype=jnp.float64)
    n_atoms = int(pos.shape[0])
    rng = np.random.default_rng(7)
    pos = pos + jnp.asarray(rng.normal(scale=0.03, size=pos.shape), dtype=jnp.float64)

    components, forces = load_monomer_bonded_components_from_psf(
        aco_psf,
        pos,
        atoms_per_monomer=n_atoms,
        energy_unit="kcal/mol",
    )
    eval_fn = load_monomer_bonded_evaluator_from_psf(
        aco_psf,
        atoms_per_monomer=n_atoms,
        energy_unit="kcal/mol",
    )
    e_total, f_eval = eval_fn(pos)
    np.testing.assert_allclose(float(e_total), float(components["total"]), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(f_eval), np.asarray(forces), rtol=1e-10, atol=1e-10)

    system = load_cgenff_bonded_from_psf(aco_psf, pos)
    mm_mask = jnp.ones(system.n_atoms, dtype=bool)
    topology, bonded, urey_k, urey_r0 = filter_bonded_topology_for_mm(
        system.topology,
        system.bonded,
        mm_mask,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
    )
    ref_comp, ref_f = bonded_energy_and_forces(
        pos,
        topology,
        bonded,
        urey_k=urey_k,
        urey_r0=urey_r0,
        energy_unit="kcal/mol",
    )
    for key in ("bond", "angle", "urey", "torsion", "improper", "total"):
        if key not in components or key not in ref_comp:
            continue
        np.testing.assert_allclose(
            float(components[key]),
            float(ref_comp[key]),
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"spoof vs cgenff component mismatch: {key}",
        )
    np.testing.assert_allclose(np.asarray(forces), np.asarray(ref_f), rtol=1e-8, atol=1e-8)
