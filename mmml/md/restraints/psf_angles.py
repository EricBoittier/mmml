"""PSF/CGenFF angle (+ optional Urey–Bradley) restraints for jax-md hybrids.

Hybrid ``ml_intra`` + ``mm_nonbonded`` liquids have no CHARMM SHAKE and no
classical bonded MM on ML monomers. When intramolecular geometry drifts
(angles leaving the tetrahedral well), these restraints restore the
PSF/CGenFF equilibrium angles without replacing the ML energy — they add a
scaled harmonic angle (and optional 1–3 Urey) term on top of the hybrid
forces.

Energy is in eV, forces in eV/Å (ASE / jax-md house units).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "PsfAngleRestraintInfo",
    "build_psf_angle_restraint_fns",
]


@dataclass(frozen=True)
class PsfAngleRestraintInfo:
    """Metadata for logging / suite summaries."""

    psf_path: str
    n_angles: int
    n_urey: int
    scale: float
    include_urey: bool
    box_A: float | None


def build_psf_angle_restraint_fns(
    psf_path: Path | str,
    positions: Any,
    *,
    box_A: float | None = None,
    scale: float = 1.0,
    include_urey: bool = True,
) -> tuple[Callable, Callable, PsfAngleRestraintInfo]:
    """Build ``(energy_fn, force_fn, info)`` from a CHARMM PSF + CGenFF PRM.

    ``energy_fn(positions) -> scalar eV`` and ``force_fn(positions) -> (N,3)``
    use only angle (+ optional Urey) terms — bonds / torsions / impropers /
    CMAP are omitted so ML can still set bond lengths.
    """
    import jax
    import jax.numpy as jnp
    from jax_md import space

    from mmml.data.units import KCAL_MOL_TO_EV
    from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_components
    from mmml.interfaces.pycharmmInterface.cgenff_topology import (
        load_cgenff_bonded_from_psf,
    )

    if float(scale) < 0.0:
        raise ValueError(f"psf angle restraint scale must be >= 0, got {scale}")

    psf_path = Path(psf_path)
    system = load_cgenff_bonded_from_psf(psf_path, np.asarray(positions, dtype=np.float64))
    topology = system.topology
    bonded = system.bonded
    urey_k = system.urey_k if include_urey else None
    urey_r0 = system.urey_r0 if include_urey else None

    if box_A is not None and float(box_A) > 0.0:
        L = float(box_A)
        displacement_fn, _ = space.periodic(jnp.asarray([L, L, L]))
    else:
        displacement_fn, _ = space.free()

    scale_f = float(scale)
    n_angles = int(np.asarray(topology.angles).shape[0])
    if urey_k is None:
        n_urey = 0
    else:
        n_urey = int(np.count_nonzero(np.asarray(urey_k)))

    def energy_eV(pos):
        comps = bonded_energy_components(
            pos,
            topology,
            bonded,
            displacement_fn,
            urey_k=urey_k,
            urey_r0=urey_r0,
            include_cmap=False,
        )
        e_kcal = comps["angle"]
        if include_urey:
            e_kcal = e_kcal + comps["urey"]
        return scale_f * e_kcal * KCAL_MOL_TO_EV

    @jax.jit
    def energy_fn(pos):
        return energy_eV(pos)

    @jax.jit
    def force_fn(pos):
        return -jax.grad(energy_eV)(pos)

    info = PsfAngleRestraintInfo(
        psf_path=str(psf_path.resolve()),
        n_angles=n_angles,
        n_urey=n_urey,
        scale=scale_f,
        include_urey=bool(include_urey),
        box_A=float(box_A) if box_A is not None else None,
    )
    return energy_fn, force_fn, info
