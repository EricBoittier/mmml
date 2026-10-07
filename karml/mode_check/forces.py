"""Force diagnostics: analytic vs finite-difference and bond stretch scans."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

# Harmonic OH-like conversion: k [eV/Å²] → ν [cm⁻¹] for reduced mass μ [amu].
_EV_TO_J = 1.602176634e-19
_A_TO_M = 1e-10
_AMU_TO_KG = 1.66053906660e-27
_C_CM_S = 2.99792458e10


def reduced_mass_amu(m1: float, m2: float) -> float:
    """Two-body reduced mass in amu."""
    a = float(m1)
    b = float(m2)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("masses must be positive")
    return a * b / (a + b)


def spring_constant_to_wavenumber_cm(
    k_eV_A2: float,
    *,
    mu_amu: float = 15.999 * 1.008 / (15.999 + 1.008),
) -> float:
    """Convert a 1D harmonic spring constant (eV/Å²) to a wavenumber (cm⁻¹)."""
    k = float(k_eV_A2)
    mu = float(mu_amu)
    if k <= 0.0 or not np.isfinite(k):
        return float("nan")
    if mu <= 0.0 or not np.isfinite(mu):
        raise ValueError("mu_amu must be positive and finite")
    omega = (1.0 / (2.0 * np.pi)) * np.sqrt(
        (k * _EV_TO_J / _A_TO_M**2) / (mu * _AMU_TO_KG)
    )
    return float(omega / _C_CM_S)


def force_fd_check(
    atoms: Atoms,
    natoms_check: int,
    dx: float,
) -> dict[str, float]:
    """Central finite-difference check of analytic ASE forces.

    Returns max abs and RMS differences over the first ``natoms_check`` atoms.
    """
    x0 = atoms.get_positions().copy()
    f_analytic = np.asarray(atoms.get_forces(), dtype=float)
    n_check = min(int(natoms_check), len(atoms))
    if n_check < 1:
        raise ValueError("natoms_check must be >= 1 for a non-empty system")
    dx = float(dx)
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    f_numeric = np.zeros((n_check, 3), dtype=float)
    for i in range(n_check):
        for a in range(3):
            xp = x0.copy()
            xm = x0.copy()
            xp[i, a] += dx
            xm[i, a] -= dx
            atoms.set_positions(xp)
            ep = float(atoms.get_potential_energy())
            atoms.set_positions(xm)
            em = float(atoms.get_potential_energy())
            f_numeric[i, a] = -(ep - em) / (2.0 * dx)
    atoms.set_positions(x0)
    _ = atoms.get_potential_energy()
    delta = f_numeric - f_analytic[:n_check, :]
    return {
        "fd_atoms_checked": float(n_check),
        "fd_dx_A": float(dx),
        "fd_force_max_abs_diff_eVA": float(np.max(np.abs(delta))),
        "fd_force_rms_diff_eVA": float(np.sqrt(np.mean(delta**2))),
    }


def stretch_force_projection(
    atoms: Atoms,
    atom_i: int,
    atom_j: int,
) -> tuple[float, float]:
    """Project relative force onto the unit bond vector i→j; return (F_stretch, r)."""
    f = np.asarray(atoms.get_forces(), dtype=float)
    r_vec = atoms.positions[atom_j] - atoms.positions[atom_i]
    r = float(np.linalg.norm(r_vec))
    if r <= 0.0:
        raise ValueError("zero bond length")
    u = r_vec / r
    f_stretch = float(np.dot(f[atom_j] - f[atom_i], u))
    return f_stretch, r


def fit_quadratic_k_from_energy(
    deltas: np.ndarray,
    energies: np.ndarray,
    *,
    fit_abs_delta_max: float = 0.03,
) -> float:
    """Fit E ≈ E0 + (1/2) k δ² near equilibrium; return k in eV/Å²."""
    d = np.asarray(deltas, dtype=float)
    e = np.asarray(energies, dtype=float)
    mask = np.abs(d) <= float(fit_abs_delta_max)
    if int(mask.sum()) < 3:
        raise ValueError("need at least 3 points inside fit window")
    a = np.vstack([np.ones(int(mask.sum())), d[mask] ** 2]).T
    coef, *_ = np.linalg.lstsq(a, e[mask], rcond=None)
    return 2.0 * float(coef[1])


def fit_k_from_force(
    deltas: np.ndarray,
    forces: np.ndarray,
    *,
    fit_abs_delta_max: float = 0.03,
) -> float:
    """Fit F ≈ −k δ near equilibrium; return k in eV/Å²."""
    d = np.asarray(deltas, dtype=float)
    f = np.asarray(forces, dtype=float)
    mask = np.abs(d) <= float(fit_abs_delta_max)
    if int(mask.sum()) < 2:
        raise ValueError("need at least 2 points inside fit window")
    dd = d[mask]
    ff = f[mask]
    denom = float(np.dot(dd, dd))
    if denom <= 0.0:
        raise ValueError("degenerate force fit")
    return -float(np.dot(dd, ff) / denom)


def bond_stretch_scan(
    atoms: Atoms,
    atom_i: int,
    atom_j: int,
    *,
    deltas: np.ndarray | None = None,
    fit_abs_delta_max: float = 0.03,
    mu_amu: float | None = None,
) -> dict[str, Any]:
    """Displace atom_j along the bond and record E / F_stretch; fit harmonic k.

    Restores the original positions before returning.
    """
    if deltas is None:
        deltas = np.linspace(-0.08, 0.08, 17)
    deltas = np.asarray(deltas, dtype=float)
    r0 = atoms.positions.copy()
    vec = r0[atom_j] - r0[atom_i]
    n0 = float(np.linalg.norm(vec))
    if n0 <= 0.0:
        raise ValueError("zero bond length at start of stretch scan")
    u = vec / n0
    rows: list[dict[str, float]] = []
    for d in deltas:
        pos = r0.copy()
        pos[atom_j] = r0[atom_i] + u * (n0 + float(d))
        atoms.set_positions(pos)
        energy = float(atoms.get_potential_energy())
        f_s, r = stretch_force_projection(atoms, atom_i, atom_j)
        rows.append(
            {
                "delta_A": float(d),
                "r_A": float(r),
                "E_eV": energy,
                "F_stretch_eV_A": float(f_s),
            }
        )
    atoms.set_positions(r0)

    d_arr = np.array([row["delta_A"] for row in rows], dtype=float)
    e_arr = np.array([row["E_eV"] for row in rows], dtype=float)
    f_arr = np.array([row["F_stretch_eV_A"] for row in rows], dtype=float)
    k_e = fit_quadratic_k_from_energy(d_arr, e_arr, fit_abs_delta_max=fit_abs_delta_max)
    k_f = fit_k_from_force(d_arr, f_arr, fit_abs_delta_max=fit_abs_delta_max)
    if mu_amu is None:
        masses = atoms.get_masses()
        mu_amu = reduced_mass_amu(float(masses[atom_i]), float(masses[atom_j]))
    return {
        "atom_i": int(atom_i),
        "atom_j": int(atom_j),
        "mu_amu": float(mu_amu),
        "rows": rows,
        "k_from_E_eV_A2": float(k_e),
        "k_from_F_eV_A2": float(k_f),
        "nu_cm_from_E": spring_constant_to_wavenumber_cm(k_e, mu_amu=float(mu_amu)),
        "nu_cm_from_F": spring_constant_to_wavenumber_cm(k_f, mu_amu=float(mu_amu)),
    }
