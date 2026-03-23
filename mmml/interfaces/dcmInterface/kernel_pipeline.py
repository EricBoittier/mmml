"""
Pipeline: optimize charge positions -> fit kernel (distance matrix -> AQ,BQ,CQ) -> write CHARMM kernel + H5.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .dcmnet_to_mdcm import _compute_charges_per_frame
from .evaluate_h5 import evaluate_and_write_h5
from .kernel_fit import (
    fit_kernel_from_training_data,
    compute_fit_metrics,
    _cpf_to_abc_flat,
)
from .optimize import optimize_charge_positions
from .topology import get_frames_meoh_like


def run_kernel_fit_pipeline(
    h5_path: Union[str, Path],
    out_dir: Union[str, Path],
    natmk: int,
    out_h5: Optional[Union[str, Path]] = None,
    optimize_positions: bool = False,
    train_frame_indices: Optional[Sequence[int]] = None,
    lam: float = 1e-6,
    sigma: float = 1.0,
    base_name: str = "x_fit",
) -> dict:
    """
    Full pipeline: (optional) optimize -> fit kernel -> write CHARMM files -> evaluate H5.

    Workflow:
    1. Load R, Z, dcmnet_charges, dcmnet_charge_positions, esp_grid, esp_reference from H5
    2. If optimize_positions: optimize (AQ,BQ,CQ) per training frame to minimize ESP RMSE
    3. Fit kernel ridge: distance matrix (upper triangle) -> (AQ,BQ,CQ) for each charge
    4. Write x_fit.txt, coefs0.txt, ... to out_dir (CHARMM KERN format)
    5. If out_h5: predict for all frames, compute ESP, write H5 for GUI

    Parameters
    ----------
    h5_path : path
        charmm_ml_comparison.h5 or similar (R, Z, N, dcmnet_*, esp_grid, esp_reference)
    out_dir : path
        Directory for x_fit.txt, coefs*.txt
    natmk : int
        Number of atoms for distance matrix (e.g. 6 for benzene)
    out_h5 : path, optional
        If set, evaluate kernel on all frames and write H5 for GUI
    optimize_positions : bool
        If True, optimize (AQ,BQ,CQ) per training frame before fitting kernel
    train_frame_indices : sequence, optional
        Frames to use for kernel training. None = all.
    lam, sigma : float
        Kernel ridge params
    base_name : str
        Prefix for x_fit/coefs filenames

    Returns
    -------
    dict with keys: X_fit, alphas, paths, out_h5_path (if out_h5 set)
    """
    import h5py

    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    if not h5_path.exists():
        raise FileNotFoundError(str(h5_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_total = f["R"].shape[0]
        has_n = "N" in f
        esp_grid = np.asarray(f["esp_grid"])
        esp_reference = np.asarray(f["esp_reference"])

    if train_frame_indices is None:
        train_frame_indices = list(range(n_total))

    # Load training data
    R_list = []
    charges_per_frame_list = []
    with h5py.File(h5_path, "r") as f:
        R_ref = np.asarray(f["R"][train_frame_indices[0]])
        Z_ref = np.asarray(f["Z"][train_frame_indices[0]])
        n_atoms = int(f["N"][train_frame_indices[0]]) if has_n else R_ref.shape[0]
        frames = get_frames_meoh_like(R_ref[:n_atoms], Z_ref[:n_atoms])
        dq_per_charge = []  # same for all frames
        for idx in train_frame_indices:
            R = np.asarray(f["R"][idx])
            Z = np.asarray(f["Z"][idx])
            charges = np.asarray(f["dcmnet_charges"][idx])
            positions = np.asarray(f["dcmnet_charge_positions"][idx])
            n = int(f["N"][idx]) if has_n else R.shape[0]
            R, Z, charges, positions = R[:n], Z[:n], charges[:n], positions[:n]
            cpf = _compute_charges_per_frame(R, charges, positions, frames)
            R_list.append(R)
            if not dq_per_charge:
                dq_per_charge = [[dq for _, _, _, dq in fc] for fc in cpf]
            if optimize_positions:
                esp_g = esp_grid[idx] if esp_grid.ndim == 3 else esp_grid
                esp_t = esp_reference[idx]
                cpf = optimize_charge_positions(R, frames, cpf, esp_g, esp_t)
            charges_per_frame_list.append(cpf)

    # Fit kernel
    X_fit, alphas, paths = fit_kernel_from_training_data(
        R_list,
        charges_per_frame_list,
        natmk=natmk,
        out_dir=out_dir,
        base_name=base_name,
        lam=lam,
        sigma=sigma,
    )

    # Fit metrics: predicted vs target (AQ, BQ, CQ) on training data
    Y_target = np.array([_cpf_to_abc_flat(cpf) for cpf in charges_per_frame_list]).T
    fit_metrics = compute_fit_metrics(X_fit, Y_target, alphas, sigma=sigma)

    result = {
        "X_fit": X_fit,
        "alphas": alphas,
        "paths": paths,
        "fit_metrics": fit_metrics,
    }

    if out_h5 is not None:
        out_h5 = Path(out_h5)
        _, esp_metrics = evaluate_and_write_h5(
            h5_src=h5_path,
            h5_out=out_h5,
            X_fit=X_fit,
            alphas=alphas,
            dq_per_charge=dq_per_charge,
            natmk=natmk,
            frames=frames,
            sigma=sigma,
            frame_indices=None,
        )
        result["out_h5_path"] = out_h5
        result["esp_metrics"] = esp_metrics

    return result
