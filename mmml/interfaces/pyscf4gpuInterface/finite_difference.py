"""
Central finite-difference nuclear gradients for PySCF-backed energy functions.

Shared by MP2 (and optionally other methods) when analytic gradients are
unavailable in gpu4pyscf.
"""

from __future__ import annotations

import numpy as np

from mmml.data.units import EV_ANGSTROM_TO_HARTREE_BOHR, HARTREE_TO_EV


def central_difference_gradient(
    energy_fn,
    positions_ang: np.ndarray,
    *,
    step_ang: float = 1e-3,
    verbose: int = 0,
) -> np.ndarray:
    """
    Compute ∂E/∂R via central differences (Cartesian, Angstrom).

    Parameters
    ----------
    energy_fn
        Callable ``(R_ang) -> float`` returning energy in Hartree.
    positions_ang
        Shape ``(natom, 3)`` in Angstrom.
    step_ang
        Displacement in Angstrom for central differences.

    Returns
    -------
    np.ndarray
        Gradient in Hartree/Bohr, shape ``(natom, 3)`` (PySCF convention).
    """
    r0 = np.asarray(positions_ang, dtype=np.float64).copy()
    natom, ndim = r0.shape
    if ndim != 3:
        raise ValueError(f"positions_ang must be (natom, 3), got {r0.shape}")
    if step_ang <= 0:
        raise ValueError(f"step_ang must be positive, got {step_ang}")

    grad_ev_ang = np.zeros_like(r0)
    for i in range(natom):
        for j in range(3):
            r_plus = r0.copy()
            r_minus = r0.copy()
            r_plus[i, j] += step_ang
            r_minus[i, j] -= step_ang
            e_plus = float(energy_fn(r_plus))
            e_minus = float(energy_fn(r_minus))
            # dE/dR in Hartree/Angstrom -> convert to eV/Ang for intermediate, then to Ha/Bohr
            de_ha_ang = (e_plus - e_minus) / (2.0 * step_ang)
            de_ev_ang = de_ha_ang * HARTREE_TO_EV
            grad_ev_ang[i, j] = de_ev_ang
            if verbose >= 2:
                print(f"  atom {i} dim {j}: dE/dR = {de_ha_ang:.6e} Ha/Å")

    # eV/Å -> Hartree/Bohr
    grad_ha_bohr = grad_ev_ang * EV_ANGSTROM_TO_HARTREE_BOHR
    return grad_ha_bohr


def gradient_to_forces_hartree_bohr(gradient_ha_bohr: np.ndarray) -> np.ndarray:
    """Forces = -gradient (Hartree/Bohr)."""
    return -np.asarray(gradient_ha_bohr, dtype=np.float64)
