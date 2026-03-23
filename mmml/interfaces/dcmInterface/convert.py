"""Coordinate conversion between global and DCM local frame."""

from __future__ import annotations

import numpy as np


def global_to_local(
    global_pos: np.ndarray,
    atom_pos: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, float, float]:
    """
    Convert global charge position to local frame coefficients (AQ, BQ, CQ).

    offset = global_pos - atom_pos
    AQ = dot(offset, X), BQ = dot(offset, Y), CQ = dot(offset, Z)

    Parameters
    ----------
    global_pos : np.ndarray
        Shape (3,)
    atom_pos : np.ndarray
        Shape (3,)
    X, Y, Z : np.ndarray
        Unit vectors, each shape (3,)

    Returns
    -------
    tuple
        (AQ, BQ, CQ)
    """
    offset = np.asarray(global_pos, dtype=float).ravel() - np.asarray(atom_pos, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    aq = float(np.dot(offset, X))
    bq = float(np.dot(offset, Y))
    cq = float(np.dot(offset, Z))
    return (aq, bq, cq)


def local_to_global(
    atom_pos: np.ndarray,
    aq: float,
    bq: float,
    cq: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Convert local frame coefficients to global position.

    global_pos = atom_pos + AQ*X + BQ*Y + CQ*Z

    Parameters
    ----------
    atom_pos : np.ndarray
        Shape (3,)
    aq, bq, cq : float
        Local frame coefficients
    X, Y, Z : np.ndarray
        Unit vectors, each shape (3,)

    Returns
    -------
    np.ndarray
        Shape (3,) global position
    """
    atom_pos = np.asarray(atom_pos, dtype=float).ravel()
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    return atom_pos + aq * X + bq * Y + cq * Z
