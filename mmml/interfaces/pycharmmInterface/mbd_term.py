"""Whole-system learned MBD dispersion correction for the hybrid ML/MM calculator.

The hybrid :func:`mmml.interfaces.pycharmmInterface.mmml_calculator.setup_calculator`
path evaluates ML per monomer/dimer and MM (CGenFF) inter-monomer. A checkpoint
trained with an additive MBD term expects::

    E = E_spooky + mbd_weight * E_mbd

where ``E_mbd`` is the *whole-system*, fully-connected learned QCML MBD energy
(same construction as :func:`mmml.models.mbd.calculator.atoms_to_mbd_batch`).
This module builds a JAX energy+force function over the real atoms so the same
correction can be added inside the hybrid calculator, matching training exactly.

MBD here is non-periodic and fully connected, mirroring how the model was
trained (cluster/gas-phase batches); it does not use the minimum-image
convention. Periodic MBD would need a different construction and is out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

# ASE unit constants, kept local so this module imports without ASE.
_BOHR_A = 0.5291772105638411  # Angstrom per Bohr (ase.units.Bohr)
_HARTREE_EV = 27.211386024367243  # eV per Hartree (ase.units.Hartree)

ANGSTROM_TO_BOHR = 1.0 / _BOHR_A
HARTREE_TO_EV = _HARTREE_EV
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = _HARTREE_EV / _BOHR_A

MBDEnergyForceFn = Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]


def build_mbd_energy_force_fn(
    checkpoint: str | Path,
    *,
    weight: float = 1.0,
    charge: float = 0.0,
    spin: float = 1.0,
    jit: bool = True,
) -> MBDEnergyForceFn:
    """Load an MBD checkpoint and return ``fn(positions_A, atomic_numbers)``.

    The returned function takes positions in Angstrom and atomic numbers for the
    *real* atoms of the system and returns ``(energy_eV, forces_eV_per_A)`` for
    the ``weight``-scaled MBD correction. Forces are the analytic gradient of the
    scaled energy, so ``forces == -dE/dx`` holds for the term that is actually
    added to the hybrid total.

    ``fn`` is closed over a fixed atom count (``atomic_numbers`` fixes the
    fully-connected pair list at trace time), which is the standard case for MD
    on a fixed topology. Recreate it if the atom count changes.
    """
    import e3x

    from mmml.models.mbd.calculator import load_mbd_model
    from mmml.models.mbd.model import mbd_energy_and_forces

    model, params = load_mbd_model(checkpoint)
    weight_f = float(weight)
    charge_f = float(charge)
    spin_f = float(spin)

    def _energy_forces(positions_A: jnp.ndarray, atomic_numbers: jnp.ndarray):
        positions_A = jnp.asarray(positions_A)
        atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)
        num_atoms = atomic_numbers.shape[0]
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

        positions_bohr = (positions_A * ANGSTROM_TO_BOHR).astype(jnp.float32)
        inputs = dict(
            positions=positions_bohr.reshape(-1, 3),
            atomic_numbers=atomic_numbers.reshape(-1),
            charge=jnp.asarray([charge_f], dtype=jnp.float32),
            spin=jnp.asarray([spin_f], dtype=jnp.float32),
            dst_idx=jnp.asarray(dst_idx, dtype=jnp.int32),
            src_idx=jnp.asarray(src_idx, dtype=jnp.int32),
            batch_segments=jnp.zeros(num_atoms, dtype=jnp.int32),
            atom_mask=jnp.ones(num_atoms, dtype=jnp.float32),
            edge_mask=jnp.ones(dst_idx.shape[0], dtype=jnp.float32),
            batch_size=1,
        )
        # mbd_energy_and_forces pops positions and differentiates the summed
        # energy w.r.t. it, returning (output, -gradient) in atomic units.
        output, forces_ha_bohr = mbd_energy_and_forces(model, params, **inputs)
        energy_ha = jnp.sum(output["energy"])

        energy_ev = weight_f * energy_ha * HARTREE_TO_EV
        forces_ev_a = (
            weight_f
            * jnp.asarray(forces_ha_bohr).reshape(num_atoms, 3)
            * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        )
        return energy_ev, forces_ev_a

    return jax.jit(_energy_forces) if jit else _energy_forces
