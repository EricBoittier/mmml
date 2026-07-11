"""Parity tests for extracted ``mmml.md.energy.terms`` vs. the cg_jaxmd originals.

The reference formulas are copied verbatim from ``examples/cg_jaxmd.py``
(``_dihedral_angle_rad``, ``_periodic_angle_delta_rad``, ``_phi_psi_restraint_energy``,
``end_to_end_distance``, ``smd_bias_energy``) and used as the oracle, so the
extraction is checked to be numerically faithful rather than just self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mmml.md.energy import HybridEnergy, available_terms  # noqa: E402
from mmml.md.energy.terms import (  # noqa: E402
    DihedralRestraint,
    DihedralRestraintTerm,
    RepulsiveCoreVdwTerm,
    SMDBiasTerm,
)
from mmml.data.units import KCAL_MOL_TO_EV  # noqa: E402
from mmml.md.energy.registry import EnergyContext  # noqa: E402
from mmml.md.system import MolecularSystem  # noqa: E402


# --- reference implementations (verbatim from examples/cg_jaxmd.py) ---------


def _ref_dihedral_angle_rad(r, atom_indices):
    p0, p1, p2, p3 = (r[atom_indices[k]] for k in range(4))
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / jnp.linalg.norm(b1)
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1
    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)
    return jnp.arctan2(y, x)


def _ref_periodic_delta(angle, target):
    return jnp.arctan2(jnp.sin(angle - target), jnp.cos(angle - target))


def _system(n, box=None, seed=0):
    rng = np.random.default_rng(seed)
    return MolecularSystem(
        R=rng.uniform(-5, 5, size=(n, 3)),
        Z=np.ones((n,), dtype=int),
        box=box,
        mol_id=np.arange(n),
    )


# --- registration -----------------------------------------------------------


def test_builtin_terms_register_on_import():
    assert "smd" in available_terms()
    assert "dihedral" in available_terms()
    assert "vdw_core" in available_terms()


# --- dihedral restraint -----------------------------------------------------


def test_dihedral_matches_reference():
    system = _system(8, seed=1)
    R = jnp.asarray(system.R)
    restraints = [
        DihedralRestraint(indices=(0, 1, 2, 3), target_deg=-60.0, k_ev=0.5),
        DihedralRestraint(indices=(2, 3, 4, 5), target_deg=140.0, k_ev=0.3),
    ]
    fns = DihedralRestraintTerm(restraints).make(system, EnergyContext())

    expected = 0.0
    for r in restraints:
        angle = _ref_dihedral_angle_rad(R, r.indices)
        delta = _ref_periodic_delta(angle, jnp.deg2rad(r.target_deg))
        expected += 0.5 * r.k_ev * delta * delta

    assert float(fns.jax_energy_fn(R)) == pytest.approx(float(expected), rel=1e-6)


def test_dihedral_zero_at_target():
    # A planar cis arrangement gives dihedral ~0; restrain to 0 deg -> ~0 energy.
    R = jnp.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    system = MolecularSystem(R=np.asarray(R), Z=np.ones(4, int), box=None, mol_id=np.arange(4))
    term = DihedralRestraintTerm([DihedralRestraint((0, 1, 2, 3), 0.0, 1.0)])
    e = float(term.make(system, EnergyContext()).jax_energy_fn(R))
    assert e == pytest.approx(0.0, abs=1e-10)


# --- SMD bias ---------------------------------------------------------------


@pytest.mark.parametrize("box", [None, np.diag([12.0, 12.0, 12.0])])
def test_smd_matches_reference(box):
    system = _system(6, box=box, seed=2)
    R = jnp.asarray(system.R)
    term = SMDBiasTerm(atom_i=0, atom_j=5, k_ev_per_A2=2.0, target=3.0)
    fns = term.make(system, EnergyContext())

    if box is None:
        disp = R[5] - R[0]
    else:
        from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

        disp = mic_displacement(R[0], R[5], jnp.asarray(box))
    d = jnp.sqrt(jnp.sum(disp * disp) + 1e-12)
    expected = 0.5 * 2.0 * (d - 3.0) ** 2

    assert float(fns.jax_energy_fn(R)) == pytest.approx(float(expected), rel=1e-6)


def test_smd_lambda_t_overrides_target():
    system = _system(4, seed=3)
    term = SMDBiasTerm(atom_i=0, atom_j=3, k_ev_per_A2=1.0, target=1.0)
    fn = term.make(system, EnergyContext()).jax_energy_fn
    R = jnp.asarray(system.R)
    # moving the restraint center changes the energy
    assert float(fn(R, lambda_t=1.0)) != pytest.approx(float(fn(R, lambda_t=5.0)))


def test_smd_default_target_is_build_time_cv():
    # target=None -> restraint centered at the initial CV -> zero energy at R0.
    system = _system(4, seed=4)
    term = SMDBiasTerm(atom_i=0, atom_j=3, k_ev_per_A2=1.0, target=None)
    fn = term.make(system, EnergyContext()).jax_energy_fn
    assert float(fn(jnp.asarray(system.R))) == pytest.approx(0.0, abs=1e-8)


# --- ASE face (forces via autodiff) -----------------------------------------


def test_ase_forces_match_finite_difference():
    system = _system(6, seed=5)
    term = SMDBiasTerm(atom_i=0, atom_j=5, k_ev_per_A2=2.0, target=3.0)
    contribution = term.make(system, EnergyContext()).ase_contribution

    class _Stub:
        def __init__(self, pos):
            self._pos = pos

        def get_positions(self):
            return self._pos

        def __len__(self):
            return len(self._pos)

    pos = np.asarray(system.R)
    _, forces = contribution(_Stub(pos))

    h = 1e-5
    fd = np.zeros_like(pos)
    for a in range(pos.shape[0]):
        for c in range(3):
            pp = pos.copy(); pp[a, c] += h
            pm = pos.copy(); pm[a, c] -= h
            ep, _ = contribution(_Stub(pp))
            em, _ = contribution(_Stub(pm))
            fd[a, c] = -(ep - em) / (2 * h)

    assert np.allclose(forces, fd, atol=1e-4)


# --- vdw_core (peptide-water repulsive wall) --------------------------------


def test_vdw_core_matches_reference():
    rng = np.random.default_rng(11)
    n_pep = 4
    n_water = 3
    n_atoms = n_pep + n_water * 3
    box = np.diag([10.0, 10.0, 10.0])
    system = MolecularSystem(
        R=rng.uniform(0, 10, size=(n_atoms, 3)),
        Z=np.ones(n_atoms, int),
        box=box,
        mol_id=np.arange(n_atoms),
    )
    water_indices = np.arange(n_pep, n_atoms).reshape(n_water, 3)
    eps = rng.uniform(0.02, 0.2, size=n_atoms)
    rmin_half = rng.uniform(0.8, 2.0, size=n_atoms)
    cutoff, width = 4.0, 1.5

    term = RepulsiveCoreVdwTerm(n_pep, water_indices, eps, rmin_half, cutoff, width)
    got = float(term.make(system, EnergyContext()).jax_energy_fn(jnp.asarray(system.R)))

    # reference (verbatim structure from cg_jaxmd.compute_peptide_water_core_vdw_energy)
    R = jnp.asarray(system.R)
    pep_pos = R[:n_pep]
    water_pos = R[jnp.asarray(water_indices)]
    disp = water_pos[:, None, :, :] - pep_pos[None, :, None, :]
    bd = jnp.asarray(np.diag(box))
    disp = disp - bd * jnp.round(disp / bd)
    dist = jnp.sqrt(jnp.maximum(jnp.sum(disp * disp, axis=-1), 1e-12))
    ep = jnp.sqrt(jnp.abs(
        jnp.asarray(eps)[:n_pep][None, :, None] * jnp.asarray(eps)[jnp.asarray(water_indices)][:, None, :]
    ))
    sig = jnp.asarray(rmin_half)[:n_pep][None, :, None] + jnp.asarray(rmin_half)[jnp.asarray(water_indices)][:, None, :]
    sig_r6 = (sig / jnp.maximum(dist, 1e-10)) ** 6
    repulsive = jnp.maximum(ep * (sig_r6 * sig_r6 - 2.0 * sig_r6), 0.0)
    switch_on = cutoff - width
    t = jnp.clip((dist - switch_on) / width, 0.0, 1.0)
    weights = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    expected = float(jnp.sum(weights * repulsive) * KCAL_MOL_TO_EV)

    assert got == pytest.approx(expected, rel=1e-6)
    assert got >= 0.0  # purely repulsive


def test_vdw_core_requires_box():
    system = _system(6, box=None)
    term = RepulsiveCoreVdwTerm(2, [[2, 3, 4]], [0.1] * 6, [1.0] * 6, 4.0, 1.0)
    with pytest.raises(ValueError):
        term.make(system, EnergyContext())


# --- composition ------------------------------------------------------------


def test_hybrid_composes_and_jits():
    system = _system(8, box=np.diag([15.0, 15.0, 15.0]), seed=6)
    terms = [
        SMDBiasTerm(0, 7, k_ev_per_A2=1.0, target=4.0),
        DihedralRestraintTerm([DihedralRestraint((0, 1, 2, 3), 30.0, 0.5)]),
    ]
    hybrid = HybridEnergy(terms, system, EnergyContext())
    efn = hybrid.as_jax_energy_fn()
    R = jnp.asarray(system.R)

    total = float(efn(R))
    parts = sum(float(t.make(system, EnergyContext()).jax_energy_fn(R)) for t in terms)
    assert total == pytest.approx(parts, rel=1e-6)

    # composed energy is jittable
    assert float(jax.jit(efn)(R)) == pytest.approx(total, rel=1e-6)
