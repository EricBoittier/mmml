"""Link-atom caps on a covalent ML/MM cut.

A ghost hydrogen is placed on each bond that has one ML atom and one MM atom,
a fixed distance from the ML atom along that bond. The metatomic system is the
ML atoms plus those ghosts. Forces on a ghost are mapped back onto the two real
atoms with the chain rule for

    R_L = R_QM + b * (R_MM - R_QM) / |R_MM - R_QM|

The ghost is not an MM particle: it has no charge and no Lennard-Jones site.
The CHARMM bond across the cut stays the mechanical connection between the
two regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LINK_BOND_LENGTH_A = 1.09


@dataclass(frozen=True, slots=True)
class LinkAtom:
    """One ghost hydrogen on a cut bond. Indices are into the full system."""

    qm_index: int
    mm_index: int
    bond_length_A: float = DEFAULT_LINK_BOND_LENGTH_A


def link_atom_position(
    r_qm: np.ndarray,
    r_mm: np.ndarray,
    bond_length_A: float = DEFAULT_LINK_BOND_LENGTH_A,
) -> np.ndarray:
    """Ghost position, ``bond_length_A`` from the ML atom toward the MM atom."""
    delta = np.asarray(r_mm, dtype=np.float64) - np.asarray(r_qm, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1.0e-8:
        raise ValueError("link atom requires a finite ML–MM bond")
    return np.asarray(r_qm, dtype=np.float64) + float(bond_length_A) * delta / dist


def project_link_force(
    force_on_ghost: np.ndarray,
    r_qm: np.ndarray,
    r_mm: np.ndarray,
    bond_length_A: float = DEFAULT_LINK_BOND_LENGTH_A,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F_QM, F_MM)`` for a force evaluated at the ghost position.

    ``J_QM^T F = (1-g) F + g u (u·F)`` and ``J_MM^T F = g (F - u (u·F))``,
    with ``u`` the QM→MM unit vector and ``g = b / |R_MM - R_QM|``. The two
    contributions sum to ``F``, so the ghost adds no net force of its own.
    """
    delta = np.asarray(r_mm, dtype=np.float64) - np.asarray(r_qm, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1.0e-8:
        raise ValueError("link atom requires a finite ML–MM bond")
    u = delta / dist
    g = float(bond_length_A) / dist
    force = np.asarray(force_on_ghost, dtype=np.float64)
    parallel = u * float(np.dot(u, force))
    f_qm = (1.0 - g) * force + g * parallel
    f_mm = g * (force - parallel)
    return f_qm, f_mm


def capped_ml_system(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    ml_indices: np.ndarray,
    links: tuple[LinkAtom, ...] | list[LinkAtom],
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic numbers and positions of the ML atoms followed by one H per link."""
    z = np.asarray(atomic_numbers, dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    ml = np.asarray(ml_indices, dtype=int).reshape(-1)
    ghosts_z = [1] * len(links)
    ghosts_r = [
        link_atom_position(pos[link.qm_index], pos[link.mm_index], link.bond_length_A)
        for link in links
    ]
    z_aug = np.concatenate([z[ml], np.asarray(ghosts_z, dtype=int)])
    if ghosts_r:
        pos_aug = np.concatenate([pos[ml], np.stack(ghosts_r, axis=0)], axis=0)
    else:
        pos_aug = pos[ml]
    return z_aug, pos_aug


def scatter_capped_forces(
    forces_augmented: np.ndarray,
    n_atoms: int,
    ml_indices: np.ndarray,
    links: tuple[LinkAtom, ...] | list[LinkAtom],
    positions: np.ndarray,
) -> np.ndarray:
    """Map ML + ghost forces onto the full system (ghosts are not atoms)."""
    forces = np.asarray(forces_augmented, dtype=np.float64)
    ml = np.asarray(ml_indices, dtype=int).reshape(-1)
    n_ml = int(ml.shape[0])
    if forces.shape != (n_ml + len(links), 3):
        raise ValueError(
            f"capped forces have shape {forces.shape}, expected "
            f"({n_ml + len(links)}, 3)"
        )
    out = np.zeros((int(n_atoms), 3), dtype=np.float64)
    out[ml] = forces[:n_ml]
    pos = np.asarray(positions, dtype=np.float64)
    for k, link in enumerate(links):
        f_qm, f_mm = project_link_force(
            forces[n_ml + k],
            pos[link.qm_index],
            pos[link.mm_index],
            link.bond_length_A,
        )
        out[link.qm_index] += f_qm
        out[link.mm_index] += f_mm
    return out
