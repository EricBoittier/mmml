"""DCM frame computation - mirrors CHARMM AXIS1 (BO bond-axis)."""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def compute_dcm_frame(
    R: np.ndarray,
    frame_atoms: Tuple[int, int, int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute local orthonormal frame (X, Y, Z) for each atom in a DCM frame.

    Mirrors CHARMM AXIS1 for BO (bond-axis) frames. Frame atoms are 0-based indices.

    Parameters
    ----------
    R : np.ndarray
        Atom positions, shape (n_atoms, 3)
    frame_atoms : tuple of 3 int
        (atm1_idx, atm2_idx, atm3_idx) - 0-based. atm3=atm2 for diatomic.

    Returns
    -------
    dict
        Keys are 0-based atom indices (0, 1, 2 for the 3 positions in frame).
        Values are (X, Y, Z) each shape (3,) in global coordinates.
        X,Y,Z are unit vectors; Z from bond, Y = cross(B1,B2), X = cross(Y,Z).
    """
    R = np.asarray(R, dtype=float)
    atm1, atm2, atm3 = frame_atoms

    # B1 = ATM1 - ATM2 (bond from atm2 toward atm1)
    B1 = R[atm1] - R[atm2]
    rb1 = np.linalg.norm(B1)
    if rb1 < 1e-12:
        raise ValueError(f"Degenerate bond: atoms {atm1},{atm2} coincident")
    Z1 = B1 / rb1
    Z2 = Z1  # atom2 shares atom1's Z

    if atm3 != atm2:
        # Triatomic
        B2 = R[atm3] - R[atm2]
        rb2 = np.linalg.norm(B2)
        if rb2 < 1e-12:
            raise ValueError(f"Degenerate bond: atoms {atm3},{atm2} coincident")
        Z3 = B2 / rb2

        # Y = cross(B1, B2) / |cross(B1, B2)|
        Y_vec = np.cross(B1, B2)
        rey = np.linalg.norm(Y_vec)
        if rey < 1e-12:
            raise ValueError(f"Collinear frame: atoms {atm1},{atm2},{atm3}")
        Y_vec = Y_vec / rey

        # X = EZ CROSS EY (CHARMM comment: "LOCAL X-AXIS = EZ CROSS EY")
        X1 = np.cross(Z1, Y_vec)
        X1 = X1 / np.linalg.norm(X1)
        X2 = X1
        X3 = np.cross(Z3, Y_vec)
        X3 = X3 / np.linalg.norm(X3)

        return {
            0: (X1, Y_vec, Z1),
            1: (X2, Y_vec, Z2),
            2: (X3, Y_vec, Z3),
        }
    else:
        # Diatomic: only Z, X and Y undefined (would need additional convention)
        # CHARMM uses only CQ (Z component) for diatomic
        raise ValueError("Diatomic frames not fully implemented; need X,Y convention")
