"""JAX CGenFF bonded clone as a stand-in for PhysNet in hybrid MLpot (testing).

Replaces checkpoint-based ML monomer/dimer evaluations with pure-JAX bonded
terms (and a simple inter-monomer repulsion on dimer batches) so box builds,
neighbor lists, COM switching, and calculator wiring can be exercised without
a PhysNet checkpoint or GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax_md.mm_forcefields.base import BondedParameters, Topology
from jax_md.mm_forcefields.oplsaa.topology import create_topology

from mmml.interfaces.pycharmmInterface.cgenff_bonded import (
    KCAL_MOL_TO_EV,
    build_bonded_energy_fn,
)
from mmml.interfaces.pycharmmInterface.cgenff_topology import (
    load_cgenff_bonded_from_psf,
)

BondedEvalFn = Callable[[Array], tuple[Array, Array]]


def minimal_chain_bonded_system(
    n_atoms: int,
    *,
    bond_k: float = 450.0,
    bond_r0: float = 1.54,
    angle_k: float = 60.0,
    angle_theta0: float = 1.91,
) -> tuple[Topology, BondedParameters]:
    """Harmonic chain bonded topology for synthetic toy monomers (no PSF)."""
    if n_atoms < 2:
        raise ValueError(f"n_atoms must be >= 2 for a chain, got {n_atoms}")
    bonds = jnp.stack(
        [jnp.arange(n_atoms - 1, dtype=jnp.int32), jnp.arange(1, n_atoms, dtype=jnp.int32)],
        axis=1,
    )
    angles = (
        jnp.stack(
            [
                jnp.arange(n_atoms - 2, dtype=jnp.int32),
                jnp.arange(1, n_atoms - 1, dtype=jnp.int32),
                jnp.arange(2, n_atoms, dtype=jnp.int32),
            ],
            axis=1,
        )
        if n_atoms >= 3
        else jnp.zeros((0, 3), dtype=jnp.int32)
    )
    topology = create_topology(
        n_atoms=n_atoms,
        bonds=bonds,
        angles=angles,
        torsions=jnp.zeros((0, 4), dtype=jnp.int32),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        molecule_id=jnp.zeros(n_atoms, dtype=jnp.int32),
    )
    bonded = BondedParameters(
        bond_k=jnp.full((n_atoms - 1,), float(bond_k)),
        bond_r0=jnp.full((n_atoms - 1,), float(bond_r0)),
        angle_k=jnp.full((angles.shape[0],), float(angle_k)),
        angle_theta0=jnp.full((angles.shape[0],), float(angle_theta0)),
        torsion_k=jnp.zeros(0),
        torsion_n=jnp.zeros(0, dtype=jnp.int32),
        torsion_gamma=jnp.zeros(0),
        improper_k=jnp.zeros(0),
        improper_n=jnp.zeros(0, dtype=jnp.int32),
        improper_gamma=jnp.zeros(0),
        cmap_maps=None,
    )
    return topology, bonded


def load_monomer_bonded_evaluator_from_psf(
    psf_path: Path | str,
    *,
    atoms_per_monomer: int,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    """Bonded evaluator for one monomer slice from a cluster PSF."""
    psf = Path(psf_path).expanduser().resolve()
    n_mono = int(atoms_per_monomer)
    positions = jnp.zeros((max(n_mono, 1), 3), dtype=jnp.float64)
    system = load_cgenff_bonded_from_psf(psf, positions)
    if system.n_atoms < n_mono:
        raise ValueError(
            f"PSF {psf} has {system.n_atoms} atoms; need at least {n_mono} for one monomer"
        )
    from mmml.interfaces.pycharmmInterface.cgenff_topology import filter_bonded_topology_for_mm

    mm_mask = jnp.zeros(system.n_atoms, dtype=bool)
    mm_mask = mm_mask.at[:n_mono].set(True)
    topology, bonded, urey_k, urey_r0 = filter_bonded_topology_for_mm(
        system.topology,
        system.bonded,
        mm_mask,
        urey_k=system.urey_k,
        urey_r0=system.urey_r0,
    )
    bonded_fn = build_bonded_energy_fn(
        topology,
        bonded,
        urey_k=urey_k,
        urey_r0=urey_r0,
        energy_unit=energy_unit,
    )

    def evaluate(positions: Array) -> tuple[Array, Array]:
        pos = positions[:n_mono]
        components, forces = bonded_fn(pos)
        return components["total"], forces

    return evaluate


def build_minimal_chain_bonded_evaluator(
    n_atoms: int,
    *,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    topology, bonded = minimal_chain_bonded_system(n_atoms)
    bonded_fn = build_bonded_energy_fn(topology, bonded, energy_unit=energy_unit)

    def evaluate(positions: Array) -> tuple[Array, Array]:
        components, forces = bonded_fn(positions)
        return components["total"], forces

    return evaluate


def _inter_monomer_soft_repulsion(
    positions: Array,
    n_a: int,
    *,
    epsilon_ev: float = 0.01,
    sigma_a: float = 3.4,
) -> tuple[Array, Array]:
    """Vacuum r^-12 repulsion between atoms in monomer A vs monomer B."""
    na = int(n_a)
    n = int(positions.shape[0])
    nb = n - na
    if nb <= 0:
        return jnp.array(0.0, dtype=positions.dtype), jnp.zeros_like(positions)
    pos_a = positions[:na]
    pos_b = positions[na : na + nb]
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    r2 = jnp.sum(diff * diff, axis=-1) + 1e-8
    r6 = r2**3
    r12 = r6 * r6
    sig6 = float(sigma_a) ** 6
    sig12 = sig6 * sig6
    pair_e = float(epsilon_ev) * sig12 / r12
    energy = jnp.sum(pair_e)
  # force from -dE/dr on each atom
    coeff = -12.0 * float(epsilon_ev) * sig12 / (r12 * r2)
    f_a = jnp.sum(coeff[:, :, None] * diff, axis=1)
    f_b = -jnp.sum(coeff[:, :, None] * diff, axis=0)
    forces = jnp.zeros_like(positions)
    forces = forces.at[:na].set(f_a)
    forces = forces.at[na : na + nb].set(f_b)
    return energy, forces


def build_jax_mm_spoof_batch_apply(
    *,
    atoms_per_monomer: int,
    max_atoms: int,
    monomer_eval: BondedEvalFn,
) -> Callable[[Array, Array, Array], dict[str, Array]]:
    """Return ``apply_model(Z, R, N)`` compatible with MLpot batching."""
    mono_n = int(atoms_per_monomer)
    mono_n_j = jnp.asarray(mono_n, dtype=jnp.int32)

    def _eval_one(R: Array, N: Array) -> tuple[Array, Array]:
        e_mono, f_mono = monomer_eval(R[:mono_n])
        mono_forces = jnp.zeros_like(R).at[:mono_n].set(f_mono)

        e_a, f_a = monomer_eval(R[:mono_n])
        e_b, f_b = monomer_eval(R[mono_n : 2 * mono_n])
        dimer_pos = R[: 2 * mono_n]
        e_nb, f_nb = _inter_monomer_soft_repulsion(dimer_pos, mono_n)
        dimer_forces = jnp.zeros_like(R)
        dimer_forces = dimer_forces.at[:mono_n].set(f_a + f_nb[:mono_n])
        dimer_forces = dimer_forces.at[mono_n : 2 * mono_n].set(f_b + f_nb[mono_n : 2 * mono_n])
        dimer_energy = e_a + e_b + e_nb

        is_dimer = N > mono_n_j

        def _take_dimer(_):
            return dimer_energy, dimer_forces

        def _take_mono(_):
            return e_mono, mono_forces

        return jax.lax.cond(is_dimer, _take_dimer, _take_mono, operand=None)

    vmapped = jax.vmap(_eval_one, in_axes=(0, 0))

    def apply_model(
        atomic_numbers: Array,
        positions: Array,
        batch_n: Array,
    ) -> dict[str, Array]:
        _ = atomic_numbers  # Z unused; topology is fixed in spoof mode
        batch_size = positions.shape[0] // max_atoms
        R = positions.reshape(batch_size, max_atoms, 3)
        N = jnp.asarray(batch_n, dtype=jnp.int32).reshape(batch_size)
        energies, forces = vmapped(R, N)
        return {
            "energy": energies.reshape(batch_size),
            "forces": forces.reshape(batch_size * max_atoms, 3),
        }

    return apply_model


def resolve_monomer_bonded_evaluator(
    *,
    atoms_per_monomer: int,
    monomer_psf: Path | str | None = None,
    energy_unit: str = "eV",
) -> BondedEvalFn:
    if monomer_psf is not None:
        return load_monomer_bonded_evaluator_from_psf(
            monomer_psf,
            atoms_per_monomer=atoms_per_monomer,
            energy_unit=energy_unit,
        )
    return build_minimal_chain_bonded_evaluator(atoms_per_monomer, energy_unit=energy_unit)


def jax_mm_spoof_enabled(args: object | None) -> bool:
    if args is None:
        return False
    if bool(getattr(args, "jax_mm_spoof", False)):
        return True
    mode = str(getattr(args, "ml_potential_mode", "") or "").strip().lower()
    return mode in {"jax_mm_clone", "jax-mm-clone", "jax_mm_spoof"}


__all__ = [
    "KCAL_MOL_TO_EV",
    "BondedEvalFn",
    "build_jax_mm_spoof_batch_apply",
    "build_minimal_chain_bonded_evaluator",
    "jax_mm_spoof_enabled",
    "load_monomer_bonded_evaluator_from_psf",
    "minimal_chain_bonded_system",
    "resolve_monomer_bonded_evaluator",
]
