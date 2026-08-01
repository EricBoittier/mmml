"""λ-dynamics neighbor plumbing.

Two regressions are pinned here:

1. ``wrapped_force_fn`` used to read the pair arrays out of ``pbc_state``
   *inside* a ``@jit`` body. jit bakes trace-time constants, so every
   ``_refresh_pbc_neighbors`` after the first was silently discarded and a
   window ran its entire trajectory on the pair list built at step 0.
2. ``_dudl_at_position`` rebuilt the neighbor list once per probe, so each
   recorded dU/dλ sample paid two full MM pair rebuilds instead of one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
# `lambda_jaxmd` imports `lambda_dynamics`, which does a module-level
# `import pycharmm.param` (lambda_dynamics.py:33). CI installs no libcharmm, so
# without this guard the import raises during *collection* -- pytest then
# reports "Interrupted: 1 error during collection" and runs zero tests, which
# fails the whole build rather than skipping one file. Nothing below actually
# needs CHARMM; deferring that import in `lambda_dynamics` would let these run
# in CI.
from tests.conftest import can_import_pycharmm

if not can_import_pycharmm():
    pytest.skip("PyCHARMM is not available", allow_module_level=True)

import jax
import jax.numpy as jnp

from mmml.cli.run.lambda_jaxmd import (
    LambdaJaxMdBundle,
    _dudl_at_position,
    _neighbor_tuple,
    _refresh_pbc_neighbors,
)


class _FakeOut:
    def __init__(self, value):
        self.ml_2b_E = jnp.asarray([value])
        self.mm_E = jnp.asarray([0.0])


def _fake_bundle(*, use_pbc=True, rebuild_counter=None):
    """Bundle whose 'energy' is just the pair-list content, so staleness shows up."""

    def spherical(pos, z, n_monomers, cutoff, *, mm_pair_idx, mm_pair_mask, box, scale=1.0):
        # Energy depends only on the pair list -> a stale list gives a stale value.
        return _FakeOut(jnp.sum(jnp.asarray(mm_pair_idx, dtype=jnp.float32)) * scale)

    state = {"box": None, "pair_idx": jnp.asarray([0.0]), "pair_mask": jnp.asarray([1.0])}

    def get_update_fn(pos, cutoff, box=None):
        def update_fn(positions, box=None):
            if rebuild_counter is not None:
                rebuild_counter["n"] += 1
            # Pair content tracks position, so a refresh is observable.
            total = float(jnp.sum(jnp.asarray(positions)))
            return jnp.asarray([total]), jnp.asarray([1.0])

        return update_fn

    return LambdaJaxMdBundle(
        wrapped_force_fn=lambda *a, **k: None,
        spherical_prod=spherical,
        spherical_on=lambda *a, **k: spherical(*a, **k, scale=1.0),
        spherical_off=lambda *a, **k: spherical(*a, **k, scale=0.25),
        shift=lambda *a, **k: None,
        masses=jnp.ones(3),
        atomic_numbers=jnp.ones(3, dtype=jnp.int32),
        n_monomers=1,
        cutoff=None,
        use_pbc=use_pbc,
        box_L=10.0,
        get_update_fn=get_update_fn,
        pbc_state=state,
    )


def test_refresh_returns_the_updated_neighbor_tuple():
    bundle = _fake_bundle()
    before = _neighbor_tuple(bundle)
    after = _refresh_pbc_neighbors(bundle, jnp.ones((3, 3)) * 2.0)

    assert after[0] is not before[0]
    assert float(after[0][0]) == pytest.approx(18.0)  # sum of 3x3 twos
    # ...and the tuple is what pbc_state now holds.
    assert after == _neighbor_tuple(bundle)


def test_refresh_is_observable_not_baked_into_a_jit_constant():
    """The bug: mutating pbc_state after tracing had no effect on forces."""
    bundle = _fake_bundle()

    @jax.jit
    def force_from_neighbor(pos, neighbor):
        pair_idx, _mask, _box = neighbor
        return jnp.sum(jnp.asarray(pair_idx))

    n1 = _refresh_pbc_neighbors(bundle, jnp.ones((3, 3)))
    f1 = float(force_from_neighbor(jnp.ones((3, 3)), n1))

    n2 = _refresh_pbc_neighbors(bundle, jnp.ones((3, 3)) * 5.0)
    f2 = float(force_from_neighbor(jnp.ones((3, 3)), n2))

    assert f1 != f2, "neighbor state must reach the compiled function as an argument"


def test_dudl_rebuilds_the_pair_list_once_per_sample_not_twice():
    counter = {"n": 0}
    bundle = _fake_bundle(rebuild_counter=counter)

    _dudl_at_position(bundle, jnp.ones((3, 3)))

    assert counter["n"] == 1, f"expected one rebuild per sample, got {counter['n']}"


def test_dudl_scores_both_probes_on_the_same_neighbor_list():
    bundle = _fake_bundle()
    # on = 1.0 * S, off = 0.25 * S with S the shared pair sum -> 0.75 * S.
    pos = jnp.ones((3, 3)) * 2.0
    dudl = _dudl_at_position(bundle, pos)
    assert dudl == pytest.approx(0.75 * 18.0)


def test_non_pbc_skips_the_rebuild_entirely():
    counter = {"n": 0}
    bundle = _fake_bundle(use_pbc=False, rebuild_counter=counter)
    _dudl_at_position(bundle, jnp.ones((3, 3)))
    assert counter["n"] == 0


def test_refresh_accepts_host_arrays_for_the_resume_path():
    """The resume path replays numpy frames from disk."""
    bundle = _fake_bundle()
    out = _refresh_pbc_neighbors(bundle, np.ones((3, 3), dtype=float))
    assert float(out[0][0]) == pytest.approx(9.0)
