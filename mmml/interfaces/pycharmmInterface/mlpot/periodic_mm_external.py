"""NumPy/MPI periodic Coulomb for the MLpot CHARMM callback (outside JAX JIT)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.periodic_mm import PeriodicMmConfig

if TYPE_CHECKING:
    pass


def read_psf_charges(n_atoms: int) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.mm_energy_forces import _get_actual_psf_charges

    chg = np.asarray(_get_actual_psf_charges(int(n_atoms)), dtype=np.float64).reshape(-1)
    if chg.shape[0] < int(n_atoms):
        raise ValueError(
            f"PSF charges length {chg.shape[0]} < n_atoms={n_atoms}"
        )
    return chg[: int(n_atoms)]


def compute_periodic_coulomb_kcalmol(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_side_A: float,
    cfg: PeriodicMmConfig,
) -> tuple[float, np.ndarray]:
    """Periodic Coulomb energy (kcal/mol) and forces (kcal/mol/Å)."""
    pos = np.asarray(positions_A, dtype=np.float64)
    chg = np.asarray(charges_e, dtype=np.float64)
    if cfg.uses_jax_pme:
        from mmml.interfaces.pycharmmInterface.long_range_backend import compute_jax_pme_coulomb

        result = compute_jax_pme_coulomb(
            pos,
            chg,
            box_length_A=float(box_side_A),
            method=cfg.jax_pme_method,
        )
        return float(result.energy_kcalmol), np.asarray(result.forces_kcalmol_A, dtype=np.float64)

    from mmml.interfaces.scafacosInterface.scafacos_session import compute_scafacos_coulomb

    result = compute_scafacos_coulomb(
        pos,
        chg,
        box_length_A=float(box_side_A),
        method=cfg.scafacos_method,
    )
    return float(result.energy_kcalmol), np.asarray(result.forces_kcalmol_A, dtype=np.float64)


def add_periodic_coulomb_to_callback(
    positions_A: np.ndarray,
    *,
    box_side_A: float,
    cfg: PeriodicMmConfig,
    energy_kcal: float,
    forces_kcal: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Add periodic Coulomb (jax-pme or ScaFaCoS) to MLpot callback totals."""
    n = int(positions_A.shape[0])
    charges = read_psf_charges(n)
    e_coul, f_coul = compute_periodic_coulomb_kcalmol(
        positions_A,
        charges,
        box_side_A=float(box_side_A),
        cfg=cfg,
    )
    forces = np.asarray(forces_kcal, dtype=np.float64).reshape(n, 3).copy()
    forces += f_coul
    return float(energy_kcal) + e_coul, forces
