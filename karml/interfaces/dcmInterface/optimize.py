"""Optimize DCM charge positions (AQ, BQ, CQ) with fixed charge magnitudes (DQ)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from . import convert, frame


def _charges_per_frame_to_flat(cpf: List[List[Tuple[float, float, float, float]]]) -> np.ndarray:
    """Flatten charges_per_frame to (n_params,) for optimization. Params = AQ, BQ, CQ only."""
    flat = []
    for frame_charges in cpf:
        for aq, bq, cq, dq in frame_charges:
            flat.extend([aq, bq, cq])
    return np.array(flat)


def _flat_to_charges_per_frame(
    x: np.ndarray,
    dq_values: List[List[float]],
) -> List[List[Tuple[float, float, float, float]]]:
    """Reconstruct charges_per_frame from flat x, using fixed DQ."""
    idx = 0
    cpf = []
    for frame_dqs in dq_values:
        frame_charges = []
        for dq in frame_dqs:
            aq, bq, cq = x[idx], x[idx + 1], x[idx + 2]
            idx += 3
            frame_charges.append((aq, bq, cq, dq))
        cpf.append(frame_charges)
    return cpf


def _cpf_to_positions_and_values(
    R: np.ndarray,
    frames: List[Tuple[int, int, int]],
    cpf: List[List[Tuple[float, float, float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert charges_per_frame to flat (positions, values) for ESP."""
    positions = []
    values = []
    for fr_idx, frame_atoms in enumerate(frames):
        fv = frame.compute_dcm_frame(R, frame_atoms)
        ca = frame_atoms[0]
        X, Y, Z_vec = fv[0]
        atom_pos = R[ca]
        for aq, bq, cq, dq in cpf[fr_idx]:
            pos = convert.local_to_global(atom_pos, aq, bq, cq, X, Y, Z_vec)
            positions.append(pos)
            values.append(dq)
    return np.array(positions), np.array(values)


def optimize_charge_positions(
    R: np.ndarray,
    frames: List[Tuple[int, int, int]],
    charges_per_frame: List[List[Tuple[float, float, float, float]]],
    esp_grid: np.ndarray,
    esp_target: np.ndarray,
    maxiter: int = 500,
) -> List[List[Tuple[float, float, float, float]]]:
    """
    Optimize (AQ, BQ, CQ) to minimize ESP RMSE, keeping DQ fixed.

    Parameters
    ----------
    R : np.ndarray
        Atom positions (n_atoms, 3)
    frames : list of (int, int, int)
        Frame definitions
    charges_per_frame : list of list of (AQ, BQ, CQ, DQ)
        Initial values; DQ will be held fixed
    esp_grid : np.ndarray
        (ngrid, 3) VdW surface or ESP grid
    esp_target : np.ndarray
        (ngrid,) reference ESP in Hartree/e
    maxiter : int
        Max optimizer iterations

    Returns
    -------
    Optimized charges_per_frame with same structure
    """
    dq_values = [[dq for _, _, _, dq in fc] for fc in charges_per_frame]
    x0 = _charges_per_frame_to_flat(charges_per_frame)
    valid = ~np.isnan(esp_target) & ~np.isnan(esp_grid).any(axis=1)
    esp_grid_valid = esp_grid[valid]
    esp_target_valid = esp_target[valid]

    def loss(x: np.ndarray) -> float:
        from ...utils.electrostatics import compute_esp_from_distributed_charges
        cpf = _flat_to_charges_per_frame(x, dq_values)
        pos, vals = _cpf_to_positions_and_values(R, frames, cpf)
        esp_pred = compute_esp_from_distributed_charges(vals, pos, esp_grid_valid)
        return float(np.mean((esp_pred - esp_target_valid) ** 2))

    res = minimize(loss, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    return _flat_to_charges_per_frame(res.x, dq_values)
