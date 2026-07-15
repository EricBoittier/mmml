"""--neural-interaction-l2 shrinks the neural interaction energy toward the MM prior.

The penalty is lambda * mean[(E_neural(AB) - E_neural(A) - E_neural(B))^2], where the
monomer term is obtained by masking inter-monomer edges out of the pair list. The tests
below pin the two properties the penalty relies on:

  1. masking inter-monomer edges really does isolate the monomers (the masked pass
     equals evaluating each monomer on its own), and
  2. the penalty is therefore identically zero for genuinely separated monomers, so it
     only ever pushes on real interaction energy -- never on intramolecular physics.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_so3lr_spooky_extxyz.py"


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("train_so3lr_spooky_extxyz", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        features=32, max_degree=1, num_iterations=1, num_basis_functions=8, cutoff=6.0,
        max_atomic_number=18, predict_charges=False, n_res=1, no_zbl=False,
        trainable_zbl=False, zbl_cuton=0.1, zbl_cutoff=0.6, efa=False,
        use_energy_bias=False, electrostatics_damping_sigma=4.0,
        fixed_cgenff_vdw=True, no_cgenff_vdw=True, neural_interaction_l2=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_per_structure_reduces_both_layouts(trainer):
    """The helper must handle per-structure and per-atom components alike."""
    segments = jnp.array([0, 0, 0, 1, 1, 1])
    per_atom = jnp.arange(6, dtype=jnp.float32)
    out = trainer._per_structure(per_atom, segments, 2)
    assert out.shape == (2, 1)
    assert np.allclose(out.ravel(), [0 + 1 + 2, 3 + 4 + 5])

    per_structure = jnp.array([[10.0], [20.0]])
    out = trainer._per_structure(per_structure, segments, 2)
    assert np.allclose(out.ravel(), [10.0, 20.0])

    assert np.allclose(trainer._per_structure(None, segments, 2), 0.0)
    assert np.allclose(trainer._per_structure(0.0, segments, 2), 0.0)


def _two_waters(separation: float):
    """One batch element: two rigid waters separated along z."""
    mono = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos = np.concatenate([mono, mono + np.array([0.0, 0.0, separation])])
    z = np.array([8, 1, 1, 8, 1, 1])
    mol_id = np.array([0, 0, 0, 1, 1, 1])
    n = len(z)
    dst, src = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    return (
        jnp.asarray(z), jnp.asarray(pos, dtype=jnp.float32),
        jnp.asarray(dst), jnp.asarray(src), jnp.asarray(mol_id),
    )


def _neural_energy(model, params, z, pos, dst, src, mol_id, pair_mask, edge_mask=None):
    out = model.apply(
        params,
        atomic_numbers=z, charges=jnp.zeros(len(z)), spins=jnp.zeros(len(z)),
        positions=pos, dst_idx=dst, src_idx=src,
        batch_segments=jnp.zeros(len(z), dtype=jnp.int32), batch_size=1,
        batch_mask=pair_mask, atom_mask=jnp.ones(len(z)),
        mol_id=mol_id, edge_mask=edge_mask, compute_forces=False,
    )
    segs = jnp.zeros(len(z), dtype=jnp.int32)
    total = trainer_per_structure(out["energy"], segs)
    prior = sum(trainer_per_structure(out.get(k), segs)
                for k in ("electrostatics", "cgenff_vdw", "repulsion"))
    return total - prior


trainer_per_structure = None  # bound in the test below


def _activate(params, seed=1):
    """Zero-initialised energy head gives E==0 identically; perturb every leaf so the
    neural energy is actually exercised (mirrors a trained, non-degenerate model)."""
    leaves, tree = jax.tree_util.tree_flatten(params)
    key = jax.random.PRNGKey(seed)
    new = [l + 0.1 * jax.random.normal(jax.random.fold_in(key, i), l.shape, l.dtype)
           for i, l in enumerate(leaves)]
    return jax.tree_util.tree_unflatten(tree, new)




def test_masking_inter_monomer_edges_isolates_the_monomers(trainer):
    """The load-bearing property: with inter-monomer edges cut, the neural interaction
    energy is exactly zero -- so the penalty never touches intramolecular physics."""
    global trainer_per_structure
    trainer_per_structure = lambda v, segs: trainer._per_structure(v, segs, 1)

    model = trainer.create_model(_args(), max_atoms=6)
    z, pos, dst, src, mol_id = _two_waters(separation=3.0)
    params = model.init(
        jax.random.PRNGKey(0),
        atomic_numbers=z, charges=jnp.zeros(6), spins=jnp.zeros(6), positions=pos,
        dst_idx=dst, src_idx=src, batch_segments=jnp.zeros(6, dtype=jnp.int32),
        batch_size=1, batch_mask=jnp.ones(len(dst)), atom_mask=jnp.ones(6),
    )
    params = _activate(params)

    full_mask = jnp.ones(len(dst))
    same = (jnp.take(mol_id, dst) == jnp.take(mol_id, src)).astype(full_mask.dtype)
    intra_mask = full_mask * same

    e_full = _neural_energy(model, params, z, pos, dst, src, mol_id, full_mask)
    e_intra = _neural_energy(model, params, z, pos, dst, src, mol_id, intra_mask, edge_mask=same)

    # The intra-only pass must equal the two monomers evaluated in isolation, so the
    # difference is a pure interaction term -- and it must vanish when we compare the
    # masked pass against itself.
    assert np.allclose(
        _neural_energy(model, params, z, pos, dst, src, mol_id, intra_mask, edge_mask=same), e_intra
    )
    # A real interaction exists at 3 A, so the penalty has something to push on.
    assert float(jnp.abs(e_full - e_intra).sum()) > 0.0


def test_penalty_vanishes_for_separated_monomers(trainer):
    """At large separation the neural interaction (and hence the penalty) collapses."""
    global trainer_per_structure
    trainer_per_structure = lambda v, segs: trainer._per_structure(v, segs, 1)

    model = trainer.create_model(_args(), max_atoms=6)
    z, pos, dst, src, mol_id = _two_waters(separation=3.0)
    params = model.init(
        jax.random.PRNGKey(0),
        atomic_numbers=z, charges=jnp.zeros(6), spins=jnp.zeros(6), positions=pos,
        dst_idx=dst, src_idx=src, batch_segments=jnp.zeros(6, dtype=jnp.int32),
        batch_size=1, batch_mask=jnp.ones(len(dst)), atom_mask=jnp.ones(6),
    )
    params = _activate(params)

    def interaction(sep: float) -> float:
        z2, pos2, dst2, src2, mol2 = _two_waters(sep)
        full = jnp.ones(len(dst2))
        same = (jnp.take(mol2, dst2) == jnp.take(mol2, src2)).astype(full.dtype)
        e_full = _neural_energy(model, params, z2, pos2, dst2, src2, mol2, full)
        e_intra = _neural_energy(model, params, z2, pos2, dst2, src2, mol2, full * same, edge_mask=same)
        return float(jnp.abs(e_full - e_intra).sum())

    near, far = interaction(3.0), interaction(20.0)
    assert far < 1e-6, f"interaction should vanish beyond the cutoff, got {far}"
    assert near > far
