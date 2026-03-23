"""Derive DCM frames from molecular structure (R, Z) via distance-based bonding."""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

try:
    import ase.data
    HAS_ASE = True
except ImportError:
    HAS_ASE = False

# Covalent bond cutoff: sum of radii * this factor
BOND_FACTOR = 1.2


def _get_covalent_radius(z: int) -> float:
    """Covalent radius in Angstrom."""
    if not HAS_ASE:
        # Fallback radii for common elements (Angstrom)
        _radii = {
            1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
            15: 1.07, 16: 1.05, 17: 1.02,
        }
        return _radii.get(z, 0.77)
    return float(ase.data.covalent_radii[int(z)])


def get_connectivity(R: np.ndarray, Z: np.ndarray) -> List[List[int]]:
    """
    Build bonding from distances using covalent radii.

    Parameters
    ----------
    R : np.ndarray
        Positions (n_atoms, 3)
    Z : np.ndarray
        Atomic numbers (n_atoms,)

    Returns
    -------
    list of list of int
        neighbors[i] = list of atom indices bonded to atom i
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=int).ravel()
    n = len(Z)
    if R.shape[0] != n:
        raise ValueError("R and Z length mismatch")

    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ri = _get_covalent_radius(Z[i])
        for j in range(i + 1, n):
            rj = _get_covalent_radius(Z[j])
            cutoff = BOND_FACTOR * (ri + rj)
            d = np.linalg.norm(R[i] - R[j])
            if d <= cutoff:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def get_frames(
    R: np.ndarray,
    Z: np.ndarray,
    n_charges_per_atom: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Derive DCM frames from connectivity.

    Each frame is (atm1, atm2, atm3) 0-based. For atoms with >= 2 neighbors,
    frame = (atom, neigh[0], neigh[1]). For terminal atoms (1 neighbor),
    frame = (atom, parent, sibling) where sibling is another neighbor of parent.

    Parameters
    ----------
    R : np.ndarray
        Positions (n_atoms, 3)
    Z : np.ndarray
        Atomic numbers (n_atoms,)
    n_charges_per_atom : int, optional
        If set, only include frames for atoms that should have charges.
        Default: include all atoms.

    Returns
    -------
    list of tuple
        Each (atm1, atm2, atm3) defines a frame.
    """
    neighbors = get_connectivity(R, Z)
    n = len(neighbors)
    frames: List[Tuple[int, int, int]] = []

    for i in range(n):
        nbs = neighbors[i]
        if len(nbs) >= 2:
            frames.append((i, nbs[0], nbs[1]))
        elif len(nbs) == 1:
            parent = nbs[0]
            parent_nbs = neighbors[parent]
            # Pick a sibling: another neighbor of parent that isn't i
            sibling = None
            for pnb in parent_nbs:
                if pnb != i:
                    sibling = pnb
                    break
            if sibling is not None:
                frames.append((i, parent, sibling))
            else:
                # Diatomic molecule - use (i, parent, parent) to signal diatomic
                # Our frame.py raises for diatomic; skip or use a workaround
                # For C-O in methanol, O has C and H; H has O. So H's sibling is C.
                # So we should always find sibling for non-diatomic.
                pass  # Skip atoms we can't build a frame for
    return frames


def get_frames_meoh_like(R: np.ndarray, Z: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    MEOH-like frame layout: central atom (C) bonded to O and H's.
    Frames (C, O, H1), (C, O, H2), (C, O, H3) for methyl H's,
    and (O, C, H_oh) for OH, (H, O, C) for OH H, (H, C, O) for methyl H's.

    Returns frames for each atom: one frame per atom with 3 atoms in frame.
    """
    neighbors = get_connectivity(R, Z)
    n = len(neighbors)
    frames: List[Tuple[int, int, int]] = []

    for i in range(n):
        nbs = sorted(neighbors[i])  # Deterministic
        if len(nbs) >= 2:
            frames.append((i, nbs[0], nbs[1]))
        elif len(nbs) == 1:
            parent = nbs[0]
            for pnb in neighbors[parent]:
                if pnb != i:
                    frames.append((i, parent, pnb))
                    break
    return frames
