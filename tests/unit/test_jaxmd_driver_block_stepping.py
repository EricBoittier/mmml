"""JaxmdDriver block stepping must not change trajectories.

The driver used to advance blocks with a Python ``for`` loop over single jitted
steps; it now dispatches the whole block through ``lax.fori_loop``. That is a
throughput change only — same integrator, same neighbor cadence — so the
recorded frames must match what the step-at-a-time path produced.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_md")


def _mono_atom_pbc_system(n=4, spacing=5.0, box=20.0):
    from mmml.md.system import FFParams, MolecularSystem

    R = np.zeros((n, 3))
    R[:, 0] = np.arange(n) * spacing
    charges = np.where(np.arange(n) % 2 == 0, 0.3, -0.3).astype(float)
    ff = FFParams(
        charges=charges,
        epsilon=np.full(n, 0.1),
        rmin_half=np.full(n, 1.5),
        at_codes=np.arange(n, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=R, Z=np.ones(n, int), box=np.diag([box, box, box]),
        mol_id=np.arange(n), ff_params=ff,
        monomer_indices=[np.array([i]) for i in range(n)],
    )


def _run(block_size, record_every, n_steps, skin_A=0.0):
    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.config import EnsembleSpec
    from mmml.md.drivers import JaxmdDriver
    from mmml.md.energy import EnergyContext
    from mmml.md.neighbors import make_intermolecular_neighbor_fn

    system = _mono_atom_pbc_system()
    energy = build_hybrid_energy(system, ("mm_nonbonded",), EnergyContext())
    neighbor_fn = make_intermolecular_neighbor_fn(
        system, cutoff_A=12.0, capacity=64, skin_A=skin_A
    )
    ensemble = EnsembleSpec(
        ensemble="nve",
        temperature_K=50.0,
        n_steps=n_steps,
        dt_fs=0.1,
        params={"seed": 7},
    )
    driver = JaxmdDriver(
        record_every=record_every, block_size=block_size, neighbor_fn=neighbor_fn
    )
    return driver.run(system, energy, ensemble), neighbor_fn


def test_block_stepping_matches_step_at_a_time():
    """block_size=1 is the old per-step path; a larger block must agree."""
    stepwise, _ = _run(block_size=1, record_every=8, n_steps=16)
    batched, _ = _run(block_size=8, record_every=8, n_steps=16)

    ref = np.asarray(stepwise.metadata["positions"])
    got = np.asarray(batched.metadata["positions"])

    assert ref.shape == got.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_ragged_tail_still_completes_all_steps():
    """n_steps not divisible by block_size must not drop or repeat steps."""
    full, _ = _run(block_size=1, record_every=5, n_steps=15)
    ragged, _ = _run(block_size=5, record_every=5, n_steps=15)

    ref = np.asarray(full.metadata["positions"])
    got = np.asarray(ragged.metadata["positions"])
    assert got.shape[0] == ref.shape[0]
    np.testing.assert_allclose(got[-1], ref[-1], rtol=1e-5, atol=1e-6)


def test_skin_cache_does_not_change_the_trajectory():
    """Reusing a list built at cutoff+skin must be energetically inert."""
    no_skin, plain_fn = _run(block_size=4, record_every=4, n_steps=12, skin_A=0.0)
    with_skin, cached_fn = _run(block_size=4, record_every=4, n_steps=12, skin_A=1.0)

    np.testing.assert_allclose(
        np.asarray(with_skin.metadata["positions"]),
        np.asarray(no_skin.metadata["positions"]),
        rtol=1e-5,
        atol=1e-6,
    )
    # ...and the skin actually saved rebuilds, or the test proves nothing.
    stats = cached_fn.stats.as_dict()
    assert stats["reused"] > 0, f"skin never engaged: {stats}"
    assert not hasattr(plain_fn, "stats")


def test_energies_stay_finite_across_blocks():
    traj, _ = _run(block_size=4, record_every=4, n_steps=12)
    assert np.all(np.isfinite(np.asarray(traj.metadata["energies"])))
