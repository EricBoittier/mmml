"""Tests for the intermolecular neighbor_fn and the full assemble→driver run."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.energy.capacity import CapacityOverflow
from mmml.md.neighbors import make_intermolecular_neighbor_fn
from mmml.md.system import FFParams, MolecularSystem


def _mono_atom_system(n=4, spacing=5.0, box=20.0):
    """n single-atom 'molecules' on a line — every pair is intermolecular."""
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
    )


def test_neighbor_fn_padded_pairs_and_mask():
    system = _mono_atom_system(n=4, spacing=5.0)
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=30.0, capacity=32)
    out = fn(np.asarray(system.R), np.asarray(system.box))

    assert set(out) == {"pair_i", "pair_j", "pair_mask"}
    assert out["pair_i"].shape == (32,)
    assert out["pair_mask"].dtype == np.int8
    # 4 distinct molecules, all within cutoff -> C(4,2)=6 intermolecular pairs
    assert int(out["pair_mask"].sum()) == 6
    # padded tail is masked off
    assert out["pair_mask"][6:].sum() == 0


def test_neighbor_fn_overflow_raises():
    system = _mono_atom_system(n=4)
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=30.0, capacity=3)
    with pytest.raises(CapacityOverflow, match="intermolecular pairs"):
        fn(np.asarray(system.R), np.asarray(system.box))


def test_neighbor_fn_auto_capacity():
    system = _mono_atom_system(n=4)
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=8.0)  # capacity auto
    out = fn(np.asarray(system.R), np.asarray(system.box))
    assert int(out["pair_mask"].sum()) >= 1


def test_auto_capacity_never_exceeds_the_pairs_that_can_exist():
    """The shell estimate assumes an unbounded medium; the box is the truth.

    Padding is not free -- masked slots are still evaluated, since fixed shapes
    are what keeps the kernel jitted -- so asking for more slots than the
    system can ever fill is per-step arithmetic thrown away. With a cutoff
    comparable to the box the old estimate did exactly that.
    """
    system = _mono_atom_system(n=6, spacing=2.0, box=12.0)
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=12.0)
    out = fn(np.asarray(system.R), np.asarray(system.box))

    # Single-atom molecules, so every pair is intermolecular: C(6,2) = 15.
    assert out["pair_i"].shape[0] == 15
    assert int(out["pair_mask"].sum()) == 15


def test_auto_capacity_subtracts_intramolecular_pairs_from_the_bound():
    """The builder drops same-molecule pairs, so they cannot need slots."""
    n_mol, per_mol = 5, 3
    n = n_mol * per_mol
    R = np.zeros((n, 3))
    R[:, 0] = np.repeat(np.arange(n_mol) * 4.0, per_mol) + np.tile(
        np.arange(per_mol) * 0.5, n_mol
    )
    ff = FFParams(
        charges=np.zeros(n), epsilon=np.full(n, 0.1), rmin_half=np.full(n, 1.5),
        at_codes=np.arange(n, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(
        R=R, Z=np.ones(n, int), box=np.diag([25.0] * 3),
        mol_id=np.repeat(np.arange(n_mol), per_mol), ff_params=ff,
    )
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=25.0)
    out = fn(np.asarray(system.R), np.asarray(system.box))

    # C(15,2) = 105 total, minus 5 molecules x C(3,2) = 15 intramolecular.
    assert out["pair_i"].shape[0] == 105 - 15


def test_auto_capacity_still_pads_when_the_box_is_large():
    """The bound is inert above roughly twice the cutoff.

    That is the regime where the headroom absorbs density fluctuations, and
    this change must not touch it.
    """
    n = 8
    R = np.zeros((n, 3))
    R[:, 0] = np.arange(n) * 3.0
    ff = FFParams(
        charges=np.zeros(n), epsilon=np.full(n, 0.1), rmin_half=np.full(n, 1.5),
        at_codes=np.arange(n, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(
        R=R, Z=np.ones(n, int), box=np.diag([200.0] * 3),
        mol_id=np.arange(n), ff_params=ff,
    )
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=6.0)
    out = fn(np.asarray(system.R), np.asarray(system.box))
    live = int(out["pair_mask"].sum())
    assert out["pair_i"].shape[0] > live, "headroom disappeared in the dilute regime"


def test_auto_capacity_holds_a_dense_configuration_without_overflowing():
    """A capacity that truncates raises; this is the guard against that."""
    rng = np.random.default_rng(3)
    n_mol = 60
    box = 18.0
    R = rng.uniform(0.0, box, size=(n_mol, 3))
    ff = FFParams(
        charges=np.zeros(n_mol), epsilon=np.full(n_mol, 0.1),
        rmin_half=np.full(n_mol, 1.5), at_codes=np.arange(n_mol, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    system = MolecularSystem(
        R=R, Z=np.ones(n_mol, int), box=np.diag([box] * 3),
        mol_id=np.arange(n_mol), ff_params=ff,
    )
    fn = make_intermolecular_neighbor_fn(system, cutoff_A=12.0)
    # Squeeze hard toward the centre: a density spike past anything equilibrium
    # sampling reaches. Must not raise.
    centre = np.full(3, box / 2.0)
    for factor in (1.0, 0.8, 0.65, 0.5):
        out = fn(centre + (R - centre) * factor, np.asarray(system.box))
        assert int(out["pair_mask"].sum()) <= out["pair_i"].shape[0]


def test_end_to_end_mm_nonbonded_nve():
    """Full pipeline: FFParams system → mm_nonbonded → neighbor_fn → JaxmdDriver."""
    pytest.importorskip("jax_md")
    from mmml.md import EnsembleSpec, RunConfig, SystemSpec, assemble_and_run

    system = _mono_atom_system(n=4, spacing=5.0, box=30.0)
    neighbor_fn = make_intermolecular_neighbor_fn(system, cutoff_A=12.0, capacity=32)

    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        terms=("mm_nonbonded",),
        ensemble=EnsembleSpec(
            ensemble="nve", dt_fs=0.1, n_steps=10,
            params={"float64": True, "seed": 0},
        ),
        backend="jaxmd",
    )
    traj = assemble_and_run(cfg, system=system, neighbor_fn=neighbor_fn)

    assert traj.n_frames >= 1
    assert np.all(np.isfinite(traj.metadata["energies"]))
    assert np.all(np.isfinite(traj.metadata["positions"]))
    assert traj.metadata["steps"] == 10


def test_assemble_auto_wires_neighbor_fn():
    """mm_nonbonded declares an intermolecular NeighborRequest → auto neighbor_fn."""
    pytest.importorskip("jax_md")
    from mmml.md import EnsembleSpec, RunConfig, SystemSpec, assemble_and_run

    system = _mono_atom_system(n=4, spacing=5.0, box=30.0)
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        terms=("mm_nonbonded",),
        ensemble=EnsembleSpec(ensemble="nve", dt_fs=0.1, n_steps=5, params={"float64": True}),
        backend="jaxmd",
    )
    # no neighbor_fn passed — assemble_and_run must build one from the term's request
    traj = assemble_and_run(cfg, system=system)
    assert np.all(np.isfinite(traj.metadata["energies"]))
    assert traj.metadata["steps"] == 5
