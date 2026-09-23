"""Centroid Verlet list for sparse ML dimers: exact parity with the all-pairs selection."""

from __future__ import annotations

import os
from itertools import combinations
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.dimer_centroid_nl import (
    CentroidDimerNeighborList,
    dimer_pair_ids,
)

N_MONO = 5  # atoms per monomer
BOX = 20.0
CP = CutoffParameters(mm_switch_on=6.0, ml_switch_width=1.5)


def _positions(n_monomers: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.6, size=(N_MONO, 3))
    g = int(np.ceil(n_monomers ** (1 / 3)))
    centers = [
        (np.array([a, b, c], float) + 0.5) * (BOX / g) + rng.normal(scale=1.0, size=3)
        for a in range(g) for b in range(g) for c in range(g)
    ][:n_monomers]
    return np.concatenate([c + base + rng.normal(scale=0.05, size=base.shape) for c in centers])


def _build(n_monomers: int, n_atoms: int, r0, **setup_kw):
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    fake_mm_fn = lambda *a, **k: (jnp.array(0.0), jnp.zeros((n_atoms, 3)))
    fake_update_fn = lambda *a, **k: (jnp.zeros((1, 2), dtype=jnp.int32), jnp.ones((1,), dtype=bool))

    def fake_build_mm(*args, **kwargs):
        if kwargs.get("use_jax_md_neighbor_list", True):
            return fake_mm_fn, fake_update_fn
        return fake_mm_fn

    z = jnp.full((n_atoms,), 6, dtype=jnp.int32)
    with patch(
        "mmml.interfaces.pycharmmInterface.mmml_calculator.build_mm_energy_forces_fn",
        side_effect=fake_build_mm,
    ):
        factory = setup_calculator(
            ATOMS_PER_MONOMER=N_MONO,
            N_MONOMERS=n_monomers,
            model_restart_path=None,
            ml_potential_mode="jax_mm_clone",
            doML=True,
            doMM=False,
            doML_dimer=True,
            MAX_ATOMS_PER_SYSTEM=2 * N_MONO,
            cell=BOX,
            defer_xla_gpu_warmup=True,
            verbose=False,
            ml_sparse_dimers=True,
            **setup_kw,
        )
        _, spherical_fn, _ = factory(
            atomic_numbers=z,
            atomic_positions=jnp.asarray(r0),
            n_monomers=n_monomers,
            cutoff_params=CP,
            doML=True,
            doMM=False,
            doML_dimer=True,
            backprop=False,
            create_ase_calculator=False,
        )

    def run(R, cand=None):
        kw = {} if cand is None else {"ml_dimer_candidates": cand}
        return spherical_fn(
            atomic_numbers=z,
            positions=jnp.asarray(R),
            n_monomers=n_monomers,
            cutoff_params=CP,
            doML=True,
            doMM=False,
            doML_dimer=True,
            box=jnp.eye(3) * BOX,
            **kw,
        )

    return spherical_fn, run


def _assert_identical(a, b):
    assert int(a.ml_n_active_dimers) == int(b.ml_n_active_dimers)
    np.testing.assert_array_equal(np.asarray(a.energy), np.asarray(b.energy))
    np.testing.assert_array_equal(np.asarray(a.forces), np.asarray(b.forces))


def test_pair_ids_match_combinations_order() -> None:
    n = 9
    pairs = np.array(list(combinations(range(n), 2)))
    np.testing.assert_array_equal(dimer_pair_ids(pairs[:, 0], pairs[:, 1], n), np.arange(len(pairs)))


@pytest.mark.parametrize("use_vesin", [True, False])
def test_candidates_superset_of_active_pairs(use_vesin: bool) -> None:
    if use_vesin:
        pytest.importorskip("vesin")
    n = 27
    idx = np.arange(n * N_MONO).reshape(n, N_MONO)
    nl = CentroidDimerNeighborList(
        idx, np.ones_like(idx, bool), active_radius=6.0, skin=1.0,
        cell=np.eye(3) * BOX, use_vesin=use_vesin,
    )
    R = _positions(n, seed=3)
    cand = nl.update_numpy(R)
    ids = cand[cand < nl.n_dimers]
    assert np.all(np.diff(ids) > 0)  # sorted, unique
    C = R.reshape(n, N_MONO, 3).mean(1)
    iu, ju = np.triu_indices(n, 1)
    d = C[ju] - C[iu]
    d -= BOX * np.round(d / BOX)
    r = np.linalg.norm(d, axis=-1)
    np.testing.assert_array_equal(ids, np.nonzero(r < nl.list_cutoff)[0])
    assert nl.backend == ("vesin" if use_vesin else "numpy")


def test_centroid_nl_parity_with_reuse_and_rebuild() -> None:
    n_monomers = 27
    n_atoms = n_monomers * N_MONO
    r0 = _positions(n_monomers)
    spherical_fn, run = _build(n_monomers, n_atoms, r0, ml_max_active_dimers=200)
    nl = spherical_fn.dimer_centroid_nl
    assert isinstance(nl, CentroidDimerNeighborList)
    n_dimers = n_monomers * (n_monomers - 1) // 2
    assert nl.n_dimers == n_dimers

    rng = np.random.default_rng(7)
    R = r0.copy()
    n_active = []
    builds = []
    # small rigid centroid moves (< skin/2 total) reuse the list; a large move rebuilds
    moves = [0.0, 0.1, 0.1, 0.15, 1.5, 0.1]
    for step, amp in enumerate(moves):
        shift = rng.normal(size=(n_monomers, 1, 3))
        shift *= amp / np.maximum(np.linalg.norm(shift, axis=-1, keepdims=True), 1e-12)
        R = (R.reshape(n_monomers, N_MONO, 3) + shift).reshape(-1, 3)
        R = R + rng.normal(scale=0.01, size=R.shape)  # intramolecular jiggle
        cand = nl.update(R, box=np.full(3, BOX))
        assert cand.shape == (nl.capacity,) and nl.capacity < n_dimers
        ref, new = run(R), run(R, cand)
        _assert_identical(ref, new)
        assert float(jnp.max(jnp.abs(ref.forces))) > 1e-3
        n_active.append(int(ref.ml_n_active_dimers))
        builds.append(nl.n_builds)
    assert builds[0] == 1
    assert builds[3] == 1, builds  # reused through the small moves
    assert builds[4] == 2, builds  # rebuilt after the 1.5 Å move
    assert min(n_active) > 0


def test_centroid_nl_capacity_overflow_grows_never_drops(capsys) -> None:
    n_monomers = 27
    n_atoms = n_monomers * N_MONO
    r0 = _positions(n_monomers, seed=1)
    spherical_fn, run = _build(n_monomers, n_atoms, r0, ml_max_active_dimers=200)
    proto = spherical_fn.dimer_centroid_nl
    nl = CentroidDimerNeighborList(
        proto.monomer_idx_arr, proto._w > 0, active_radius=proto.active_radius,
        skin=proto.skin, cell=np.eye(3) * BOX, capacity=4,
    )
    nl.capacity = 4  # force the overflow path on the first build too
    cand = nl.update(r0)
    assert nl.n_capacity_grows == 1
    assert "growing to" in capsys.readouterr().out
    assert nl.capacity >= nl.n_candidates > 4
    assert int(jnp.sum(cand < nl.n_dimers)) == nl.n_candidates
    _assert_identical(run(r0), run(r0, cand))

    # denser arrangement on a later rebuild: grows again, still exact
    R = (r0.reshape(n_monomers, N_MONO, 3) * np.array([1.0, 1.0, 1.0])).reshape(-1, 3)
    C = R.reshape(n_monomers, N_MONO, 3).mean(1, keepdims=True)
    R = (C * 0.6 + (R.reshape(n_monomers, N_MONO, 3) - C)).reshape(-1, 3)
    cap_before = nl.capacity
    cand2 = nl.update(R)
    if nl.n_candidates > cap_before:
        assert nl.capacity > cap_before and nl.n_capacity_grows == 2
    _assert_identical(run(R), run(R, cand2))


def test_centroid_nl_padding_only_candidates_match_empty_active_set() -> None:
    """All-padding candidates (no pair in range) == all-pairs path with nothing active."""
    n_monomers = 8
    n_atoms = n_monomers * N_MONO
    rng = np.random.default_rng(2)
    base = rng.normal(scale=0.5, size=(N_MONO, 3))
    centers = [[x, y, z] for x in (2.5, 12.5) for y in (2.5, 12.5) for z in (2.5, 12.5)]
    r0 = np.concatenate([np.asarray(c) + base for c in centers])  # 10 Å apart
    spherical_fn, run = _build(n_monomers, n_atoms, r0, ml_max_active_dimers=10)
    nl = spherical_fn.dimer_centroid_nl
    cand = nl.update(r0)
    assert nl.n_candidates == 0
    ref, new = run(r0), run(r0, cand)
    assert int(ref.ml_n_active_dimers) == 0
    _assert_identical(ref, new)


def test_centroid_nl_opt_out(monkeypatch) -> None:
    monkeypatch.setenv("MMML_ML_DIMER_CENTROID_NL", "0")
    n_monomers = 8
    r0 = _positions(n_monomers)
    spherical_fn, _ = _build(n_monomers, n_monomers * N_MONO, r0, ml_max_active_dimers=10)
    assert spherical_fn.dimer_centroid_nl is None
