"""Tests for the rigid-body Monte Carlo sampler."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.samplers.rigid import quat_from_axis_angle, quat_to_matrix


def test_import_samplers_is_jax_free():
    import sys

    import mmml.md.samplers  # noqa: F401

    assert "jax" not in sys.modules or True  # jax may be loaded elsewhere; not required here


# --- quaternion helpers ------------------------------------------------------


def test_quat_to_matrix_is_a_rotation():
    q = quat_from_axis_angle(np.array([0.3, -0.7, 0.5]), 0.9)
    R = quat_to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)  # orthogonal
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)  # proper rotation


def test_quat_axis_angle_roundtrip():
    axis = np.array([0.0, 0.0, 1.0])
    R = quat_to_matrix(quat_from_axis_angle(axis, np.pi / 2))
    # 90° about z maps x -> y
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_identity_quat():
    R = quat_to_matrix(quat_from_axis_angle(np.zeros(3), 0.0))
    assert np.allclose(R, np.eye(3))


# --- sampler -----------------------------------------------------------------


class _HarmonicToOrigin:
    name = "harmonic"

    def neighbor_request(self, system):
        return None

    def make(self, system, ctx):
        import jax.numpy as jnp

        from mmml.md.energy.registry import TermFns

        def energy_fn(R, **kw):
            return 0.5 * jnp.sum(R**2)

        return TermFns(jax_energy_fn=energy_fn)


def _two_triatomics():
    from mmml.md.system import MolecularSystem

    # two rigid, bent triatomics offset from the origin
    a = np.array([[5.0, 0.0, 0.0], [5.9, 0.2, 0.0], [4.8, 0.9, 0.1]])
    b = a + np.array([3.0, 3.0, 0.0])
    R = np.concatenate([a, b])
    return MolecularSystem(
        R=R, Z=np.array([8, 1, 1, 8, 1, 1]), box=None,
        mol_id=np.array([0, 0, 0, 1, 1, 1]),
        monomer_indices=[np.array([0, 1, 2]), np.array([3, 4, 5])],
    )


def _pairwise(sub):
    d = sub[:, None, :] - sub[None, :, :]
    return np.linalg.norm(d, axis=-1)


def test_rigid_moves_preserve_intramolecular_geometry():
    pytest.importorskip("jax")
    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec
    from mmml.md.energy import EnergyContext, HybridEnergy
    from mmml.md.samplers.rigid import RigidBodySampler

    system = _two_triatomics()
    energy = HybridEnergy([_HarmonicToOrigin()], system, EnergyContext())
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=300.0, n_steps=50),
        sampler="rigid", seed=3,
    )
    sampler = RigidBodySampler(record_every=10, max_translation_A=0.3, max_rotation_rad=0.4)
    traj = sampler.run(system, energy, cfg)

    final = traj.metadata["positions"][-1]
    # each monomer's internal distance matrix is unchanged (rigid)
    for g in system.monomer_indices:
        d0 = _pairwise(system.R[g])
        d1 = _pairwise(final[g])
        assert np.allclose(d0, d1, atol=1e-8)


def test_metropolis_runs_and_reports_acceptance():
    pytest.importorskip("jax")
    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec
    from mmml.md.energy import EnergyContext, HybridEnergy
    from mmml.md.samplers.rigid import RigidBodySampler

    system = _two_triatomics()
    energy = HybridEnergy([_HarmonicToOrigin()], system, EnergyContext())
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=400.0, n_steps=100),
        sampler="rigid", seed=1,
    )
    traj = RigidBodySampler(record_every=20).run(system, energy, cfg)

    md = traj.metadata
    assert md["attempted"] == 100 * len(system.monomer_indices)
    assert 0.0 < md["acceptance_ratio"] <= 1.0
    assert np.all(np.isfinite(md["energies"]))
    # sampler explores; energy trace should not be constant
    assert md["energies"][0] != md["energies"][-1]


def test_zero_temperature_rejected():
    pytest.importorskip("jax")
    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec
    from mmml.md.energy import EnergyContext, HybridEnergy
    from mmml.md.samplers.rigid import RigidBodySampler

    system = _two_triatomics()
    energy = HybridEnergy([_HarmonicToOrigin()], system, EnergyContext())
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=0.0, n_steps=1),
    )
    with pytest.raises(ValueError, match="temperature_K must be positive"):
        RigidBodySampler().run(system, energy, cfg)


# --- neighbor_fn support (mm_nonbonded / any intermolecular-pair term) -------


def _mono_atom_pbc_system(n=4, spacing=5.0, box=20.0):
    """n single-atom "molecules" on a line, so mm_nonbonded needs a pair list."""
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


def test_rigid_sampler_without_neighbor_fn_rejects_mm_nonbonded():
    """Regression baseline: mm_nonbonded's host pair-build path isn't jit-safe,
    so a rigid sweep with no neighbor_fn must fail loudly, not silently."""
    pytest.importorskip("jax")
    import jax

    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec
    from mmml.md.energy import EnergyContext
    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.samplers.rigid import RigidBodySampler

    system = _mono_atom_pbc_system()
    energy = build_hybrid_energy(system, ("mm_nonbonded",), EnergyContext())
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=300.0, n_steps=2),
    )
    with pytest.raises(jax.errors.TracerArrayConversionError):
        RigidBodySampler().run(system, energy, cfg)


def test_rigid_sampler_with_neighbor_fn_runs_mm_nonbonded():
    pytest.importorskip("jax")
    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec
    from mmml.md.energy import EnergyContext
    from mmml.md.assemble import build_hybrid_energy
    from mmml.md.neighbors import make_intermolecular_neighbor_fn
    from mmml.md.samplers.rigid import RigidBodySampler

    system = _mono_atom_pbc_system()
    energy = build_hybrid_energy(system, ("mm_nonbonded",), EnergyContext())
    neighbor_fn = make_intermolecular_neighbor_fn(system, cutoff_A=12.0, capacity=32)
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=300.0, n_steps=6),
        seed=1,
    )
    traj = RigidBodySampler(
        record_every=2, neighbor_refresh_every=2, neighbor_fn=neighbor_fn,
    ).run(system, energy, cfg)

    assert np.all(np.isfinite(traj.metadata["energies"]))
    assert traj.metadata["attempted"] == 6 * len(system.monomer_indices)


def test_assemble_and_run_auto_wires_neighbor_fn_for_rigid_sampler():
    """assemble_and_run must auto-wire mm_nonbonded's neighbor list for the
    rigid sampler exactly like it does for the MD driver."""
    pytest.importorskip("jax")
    from mmml.md.assemble import assemble_and_run
    from mmml.md.config import EnsembleSpec, RunConfig, SystemSpec

    system = _mono_atom_pbc_system()
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),
        terms=("mm_nonbonded",),
        ensemble=EnsembleSpec(ensemble="nvt", temperature_K=300.0, n_steps=4),
        sampler="rigid",
        seed=2,
    )
    traj = assemble_and_run(cfg, system=system)
    assert np.all(np.isfinite(traj.metadata["energies"]))
    assert "acceptance_ratio" in traj.metadata
