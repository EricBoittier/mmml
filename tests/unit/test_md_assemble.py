"""Tests for the RunConfig assembly glue (builder registry + energy + driver)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md import (
    EnsembleSpec,
    RunConfig,
    SystemSpec,
    assemble_and_run,
    available_builders,
    build_hybrid_energy,
    get_builder,
)
from mmml.md.system import MolecularSystem


def _periodic_system(n_side: int = 3, spacing: float = 2.5) -> MolecularSystem:
    grid = np.arange(n_side) * spacing
    pts = np.array([[x, y, z] for x in grid for y in grid for z in grid], dtype=float)
    L = float(n_side * spacing)
    n = len(pts)
    return MolecularSystem(
        R=pts, Z=np.ones(n, int), box=np.diag([L, L, L]), mol_id=np.arange(n)
    )


def test_builder_registry_lists_and_resolves():
    names = available_builders()
    assert {"psf", "packmol", "pyxtal", "peptide_water"} <= set(names)
    from mmml.md.builders import PsfSystemBuilder

    assert isinstance(get_builder("psf"), PsfSystemBuilder)
    with pytest.raises(KeyError, match="Unknown builder"):
        get_builder("does_not_exist")


def test_build_hybrid_energy_with_term_kwargs():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    system = _periodic_system()
    hybrid = build_hybrid_energy(
        system,
        ("smd",),
        term_kwargs={"smd": {"atom_i": 0, "atom_j": 1, "k_ev_per_A2": 1.0, "target": 2.0}},
    )
    # composes to a callable jax energy
    e = float(hybrid.as_jax_energy_fn()(jnp.asarray(system.R)))
    assert np.isfinite(e)


def test_assemble_and_run_prebuilt_system(tmp_path):
    pytest.importorskip("jax_md")
    system = _periodic_system()
    cfg = RunConfig(
        system=SystemSpec(builder="psf"),  # not used: system passed in
        terms=("smd",),
        ensemble=EnsembleSpec(ensemble="nve", dt_fs=0.5, n_steps=10, params={"seed": 0}),
        backend="jaxmd",
        output_dir=tmp_path,
    )
    traj = assemble_and_run(
        cfg,
        system=system,
        term_kwargs={"smd": {"atom_i": 0, "atom_j": 1, "k_ev_per_A2": 0.5, "target": 2.0}},
    )
    assert traj.n_frames >= 1
    assert (tmp_path / "trajectory.npz").exists()
    assert np.all(np.isfinite(traj.metadata["energies"]))


def test_assemble_and_run_rejects_non_jaxmd_backend():
    cfg = RunConfig(system=SystemSpec(builder="psf"), backend="pycharmm")
    with pytest.raises(NotImplementedError, match="jaxmd backend"):
        assemble_and_run(cfg, system=_periodic_system())
