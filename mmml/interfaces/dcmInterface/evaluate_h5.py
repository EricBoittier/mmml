"""
Evaluate fitted/optimized DCM: compute ESP and charge positions, write H5 for GUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import convert, frame
from .kernel_fit import predict_charges_from_kernel


def _esp_metrics(esp_pred: np.ndarray, esp_ref: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R² for ESP (masking NaN)."""
    p_flat = np.asarray(esp_pred, dtype=float).ravel()
    r_flat = np.asarray(esp_ref, dtype=float).ravel()
    valid = ~np.isnan(r_flat) & ~np.isnan(p_flat)
    if not np.any(valid):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    p, r = p_flat[valid], r_flat[valid]
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    mae = float(np.mean(np.abs(p - r)))
    ss_res = np.sum((r - p) ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


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


def _cpf_to_dcmnet_shapes(
    n_atoms: int,
    n_charges_per_atom: int,
    cpf: List[List[Tuple[float, float, float, float]]],
    R: np.ndarray,
    frames: List[Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert charges_per_frame to (n_atoms, n_charges) and (n_atoms, n_charges, 3)."""
    charges = np.zeros((n_atoms, n_charges_per_atom))
    positions = np.zeros((n_atoms, n_charges_per_atom, 3))
    for fr_idx, frame_atoms in enumerate(frames):
        fv = frame.compute_dcm_frame(R, frame_atoms)
        ca = frame_atoms[0]
        X, Y, Z_vec = fv[0]
        atom_pos = R[ca]
        for c, (aq, bq, cq, dq) in enumerate(cpf[fr_idx]):
            if c < n_charges_per_atom:
                charges[ca, c] = dq
                positions[ca, c] = convert.local_to_global(
                    atom_pos, aq, bq, cq, X, Y, Z_vec
                )
    return charges, positions


def evaluate_and_write_h5(
    h5_src: Union[str, Path],
    h5_out: Union[str, Path],
    X_fit: np.ndarray,
    alphas: np.ndarray,
    dq_per_charge: List[List[float]],
    natmk: int,
    frames: Optional[List[Tuple[int, int, int]]] = None,
    sigma: float = 1.0,
    frame_indices: Optional[Sequence[int]] = None,
) -> Path:
    """
    For each conformation in h5_src: predict charges from kernel, compute ESP, write to h5_out.

    The output H5 has the same structure as charmm_ml_comparison.h5 plus:
    - esp_kernel: ESP from kernel-predicted charges
    - esp_errors_kernel: esp_kernel - esp_reference
    - dcmnet_charge_positions_kernel: kernel-predicted positions (or overwrite dcmnet_*)

    Parameters
    ----------
    h5_src : path
        Input H5 (charmm_ml_comparison or similar) with R, Z, N, esp_grid, esp_reference
    h5_out : path
        Output H5 for GUI
    X_fit, alphas : kernel fit from fit_kernel_from_training_data
    dq_per_charge : fixed DQ structure
    natmk : int
    frames : optional, from first-conformation topology
    sigma : float
    frame_indices : optional, subset of conformations to process

    Returns
    -------
    Tuple[Path, dict]
        Path to written h5_out and esp_metrics (rmse, mae, r2 for kernel vs reference)
    """
    import h5py

    h5_src = Path(h5_src)
    h5_out = Path(h5_out)
    if not h5_src.exists():
        raise FileNotFoundError(str(h5_src))

    with h5py.File(h5_src, "r") as f:
        n_total = f["R"].shape[0]
        has_n = "N" in f
        esp_grid = np.asarray(f["esp_grid"])
        esp_reference = np.asarray(f["esp_reference"])
        R_arr = np.asarray(f["R"])
        Z_arr = np.asarray(f["Z"])
        dcmnet_charges_src = np.asarray(f["dcmnet_charges"])
        N_arr = np.asarray(f["N"]) if has_n else None

    if frame_indices is None:
        frame_indices = list(range(n_total))

    from .topology import get_frames_meoh_like

    n_charges_per_atom = dcmnet_charges_src.shape[2] if dcmnet_charges_src.ndim >= 2 else 1

    esp_kernel_list = []
    dcmnet_charges_list = []
    dcmnet_positions_list = []
    frames_local = frames

    for idx in frame_indices:
        R = np.asarray(R_arr[idx])
        Z = np.asarray(Z_arr[idx])
        n_atoms = int(N_arr[idx]) if N_arr is not None else R.shape[0]
        R = R[:n_atoms]
        Z = Z[:n_atoms]

        if frames_local is None:
            frames_local = get_frames_meoh_like(R, Z)

        cpf = predict_charges_from_kernel(
            R, X_fit, alphas, dq_per_charge, natmk, sigma=sigma
        )
        charges, positions = _cpf_to_dcmnet_shapes(
            n_atoms, n_charges_per_atom, cpf, R, frames_local
        )
        pos_flat, vals_flat = _cpf_to_positions_and_values(R, frames_local, cpf)
        from ...utils.electrostatics import compute_esp_from_distributed_charges
        esp_grid_i = esp_grid[idx] if esp_grid.ndim == 3 else esp_grid
        esp_pred = compute_esp_from_distributed_charges(
            vals_flat, pos_flat, esp_grid_i
        )

        esp_kernel_list.append(esp_pred)
        dcmnet_charges_list.append(charges)
        dcmnet_positions_list.append(positions)

    esp_kernel_arr = np.array(esp_kernel_list)
    esp_ref_sub = esp_reference[np.array(frame_indices)]
    esp_errors_kernel_arr = esp_kernel_arr - esp_ref_sub
    dcmnet_charges_arr = np.array(dcmnet_charges_list)
    dcmnet_positions_arr = np.array(dcmnet_positions_list)

    # ESP metrics (flatten and mask NaN)
    esp_metrics = _esp_metrics(esp_kernel_arr, esp_ref_sub)

    # Write output - copy from src, subset by frame_indices. Overwrite dcmnet_* with
    # kernel-fitted values so GUI shows them. Add esp_kernel for ESP dropdown.
    fidx = np.array(frame_indices)
    with h5py.File(h5_src, "r") as f_src, h5py.File(h5_out, "w") as f_out:
        for key in f_src.keys():
            data = np.asarray(f_src[key][...])
            if data.shape[0] == n_total:
                data = data[fidx]
            # Use kernel-fitted charges/positions as primary for GUI
            if key == "dcmnet_charges":
                data = dcmnet_charges_arr
            elif key == "dcmnet_charge_positions":
                data = dcmnet_positions_arr
            f_out.create_dataset(key, data=data)
        f_out.create_dataset("esp_kernel", data=esp_kernel_arr)
        f_out.create_dataset("esp_errors_kernel", data=esp_errors_kernel_arr)
        # Store ESP metrics as scalar datasets for easy inspection
        f_out.create_dataset("esp_rmse_kernel", data=np.array(esp_metrics["rmse"]))
        f_out.create_dataset("esp_mae_kernel", data=np.array(esp_metrics["mae"]))
        f_out.create_dataset("esp_r2_kernel", data=np.array(esp_metrics["r2"]))

    return h5_out, esp_metrics
