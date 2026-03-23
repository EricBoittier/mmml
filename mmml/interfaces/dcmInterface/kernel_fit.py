"""Kernel regression fit: distance matrix -> (AQ, BQ, CQ) for CHARMM KERN mode."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np


def compute_distance_matrix_upper(R: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Compute upper-triangle pairwise distances for NATMK atoms.

    Order matches CHARMM DISTM_N: for I=1..NATMK, J=1..NATMK, I<J.
    XINPUT(RR, C) = dist(atom_i, atom_j) for (i,j) in upper triangle.

    Parameters
    ----------
    R : np.ndarray
        (n_atoms, 3) or padded - use first n_atoms
    n_atoms : int
        NATMK - number of atoms in kernel molecule (e.g. 6 for benzene)

    Returns
    -------
    np.ndarray
        (NKDIST,) where NKDIST = (n_atoms**2 - n_atoms)//2
    """
    R = np.asarray(R, dtype=float)[:n_atoms]
    out = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i < j:
                d = np.linalg.norm(R[i] - R[j])
                out.append(d)
    return np.array(out)


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    RBF kernel k(x, y) = exp(-||x-y||^2 / (2*sigma^2)).
    CHARMM uses sigma=1.
    """
    # X: (n_query, n_dim), Y: (n_train, n_dim)
    diff = X[:, None, :] - Y[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    return np.exp(-dist_sq / (2 * sigma**2))


def fit_kernel_ridge(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    lam: float = 1e-6,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Kernel ridge regression: Y = K @ alpha, solve (K + lam*I) alpha = Y.

    Parameters
    ----------
    X_train : np.ndarray
        (NTRAIN, NKDIST) training distance vectors
    Y_train : np.ndarray
        (NTRAIN,) or (n_outputs, NTRAIN) target values (AQ, BQ, or CQ)
    lam : float
        Ridge regularization
    sigma : float
        RBF kernel width

    Returns
    -------
    np.ndarray
        ALPHAS (n_outputs, NTRAIN) or (NTRAIN,) for single output
    """
    K = rbf_kernel(X_train, X_train, sigma)
    single = Y_train.ndim == 1
    if single:
        Y_train = Y_train.reshape(1, -1)
    n_out = Y_train.shape[0]
    alphas = np.linalg.solve(K.T + lam * np.eye(K.shape[0]), Y_train.T).T
    return alphas[0] if single else alphas


def predict_kernel(
    X_query: np.ndarray,
    X_train: np.ndarray,
    alphas: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Predict: y = K(X_query, X_train) @ alphas."""
    K = rbf_kernel(X_query, X_train, sigma)
    # alphas: (n_out, NTRAIN), K: (n_query, NTRAIN) -> (n_query, n_out)
    return K @ alphas.T


def compute_fit_metrics(
    X_fit: np.ndarray,
    Y_target: np.ndarray,
    alphas: np.ndarray,
    sigma: float = 1.0,
) -> dict:
    """
    Compute fit errors: predicted vs target (AQ, BQ, CQ).

    Returns
    -------
    dict with keys: rmse, mae, r2, rmse_per_output
    """
    Y_pred = predict_kernel(X_fit, X_fit, alphas, sigma=sigma)
    # Y_target: (n_out, NTRAIN), Y_pred: (NTRAIN, n_out)
    Y_pred = Y_pred.T  # (n_out, NTRAIN)
    diff = Y_pred - Y_target
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((Y_target - np.mean(Y_target))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse_per_output = np.sqrt(np.mean(diff**2, axis=1))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_per_output": rmse_per_output,
    }


def write_kernel_files(
    out_dir: Union[str, Path],
    X_fit: np.ndarray,
    alphas: np.ndarray,
    base_name: str = "uuid",
) -> List[Path]:
    """
    Write CHARMM kernel format: x_fit.txt, coefs0.txt, coefs1.txt, ...

    Parameters
    ----------
    out_dir : path
        Directory for output (e.g. acec8c45-bfe5-4acc-8574-925372ecb40d/)
    X_fit : np.ndarray
        (NTRAIN, NKDIST) training distance matrices
    alphas : np.ndarray
        (NKERNC*3, NTRAIN) coefficients for AQ, BQ, CQ of each charge
    base_name : str
        Prefix for subdir/filenames

    Returns
    -------
    list of Path
        Written file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    # Use CHARMM-style names: x_fit.txt, coefs0.txt when base_name is "x_fit"
    if base_name == "x_fit":
        x_path = out_dir / "x_fit.txt"
        coef_fmt = "coefs{i}.txt"
    else:
        x_path = out_dir / f"{base_name}_x_fit.txt"
        coef_fmt = f"{base_name}_coefs{{i}}.txt"
    np.savetxt(x_path, X_fit, fmt="%.8e")
    written.append(x_path)
    for i in range(alphas.shape[0]):
        coef_path = out_dir / coef_fmt.format(i=i)
        np.savetxt(coef_path, alphas[i], fmt="%.8e")
        written.append(coef_path)
    return written


def build_kernel_header_line(
    ntrain: int,
    nkernc: int,
    nkfr: int,
    natmk: int,
) -> str:
    """First line of CHARMM kernel file: NTRAIN NKERNC NKFR NATMK."""
    return f"{ntrain} {nkernc} {nkfr} {natmk}"


def _kernel_filenames(out_dir: Path, base_name: str, nkernc: int) -> List[str]:
    """Filenames for x_fit and coefs (matches write_kernel_files)."""
    if base_name == "x_fit":
        return [str(out_dir / "x_fit.txt")] + [
            str(out_dir / f"coefs{i}.txt") for i in range(nkernc * 3)
        ]
    return [str(out_dir / f"{base_name}_x_fit.txt")] + [
        str(out_dir / f"{base_name}_coefs{i}.txt") for i in range(nkernc * 3)
    ]


def build_kernel_filename_lines(
    out_dir: Union[str, Path],
    base_name: str,
    nkernc: int,
) -> List[str]:
    """Lines 2..NKERNC*3+1: paths to x_fit, coefs0, coefs1, ..."""
    out_dir = Path(out_dir).resolve()
    return _kernel_filenames(out_dir, base_name, nkernc)


def write_kmdcm(
    out_path: Union[str, Path],
    out_dir: Union[str, Path],
    base_name: str,
    ntrain: int,
    nkernc: int,
    nkfr: int,
    natmk: int,
    use_relative_paths: bool = True,
) -> Path:
    """
    Write CHARMM kernel .kmdcm file (DCM KERN input).

    Format:
        Line 1: NTRAIN NKERNC NKFR NATMK
        Lines 2..: path to x_fit.txt, coefs0.txt, coefs1.txt, ...

    Parameters
    ----------
    out_path : path
        Output .kmdcm file path
    out_dir : path
        Directory containing x_fit.txt, coefs*.txt
    base_name : str
        Filename prefix (x_fit_x_fit.txt, x_fit_coefs0.txt, ...)
    ntrain, nkernc, nkfr, natmk : int
        CHARMM kernel header values
    use_relative_paths : bool
        If True, paths in kmdcm are relative to kmdcm file location
    """
    out_path = Path(out_path)
    out_dir = Path(out_dir).resolve()
    kmdcm_dir = out_path.resolve().parent

    lines = [build_kernel_header_line(ntrain, nkernc, nkfr, natmk)]
    fnames = _kernel_filenames(out_dir, base_name, nkernc)
    if use_relative_paths and out_dir == kmdcm_dir:
        # Same dir: just filenames
        for p in fnames:
            lines.append(Path(p).name)
    elif use_relative_paths:
        # out_dir is subdir of kmdcm dir or sibling
        try:
            rel = out_dir.relative_to(kmdcm_dir)
            prefix = str(rel) + "/"
        except ValueError:
            prefix = str(out_dir) + "/"
        for p in fnames:
            lines.append(prefix + Path(p).name)
    else:
        lines.extend(fnames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def _cpf_to_abc_flat(cpf: List) -> np.ndarray:
    """Extract (AQ,BQ,CQ) from charges_per_frame as (NKERNC*3,)."""
    flat = []
    for fc in cpf:
        for aq, bq, cq, _ in fc:
            flat.extend([aq, bq, cq])
    return np.array(flat)


def _abc_flat_to_cpf(flat: np.ndarray, dq_per_charge: List[List[float]]) -> List:
    """Reconstruct charges_per_frame from flat (AQ,BQ,CQ) and fixed DQ."""
    idx = 0
    cpf = []
    for dqs in dq_per_charge:
        frame_charges = []
        for dq in dqs:
            aq, bq, cq = flat[idx], flat[idx + 1], flat[idx + 2]
            idx += 3
            frame_charges.append((aq, bq, cq, dq))
        cpf.append(frame_charges)
    return cpf


def fit_kernel_from_training_data(
    R_list: Sequence[np.ndarray],
    charges_per_frame_list: Sequence[List],
    natmk: int,
    out_dir: Union[str, Path],
    base_name: str = "x_fit",
    lam: float = 1e-6,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """
    Fit kernel ridge: distance matrix -> (AQ, BQ, CQ). Write CHARMM kernel files.

    Parameters
    ----------
    R_list : sequence of (n_atoms, 3)
        Atom positions for each training conformation
    charges_per_frame_list : sequence of charges_per_frame
        (AQ,BQ,CQ,DQ) per frame for each conformation
    natmk : int
        Number of atoms for distance matrix (first natmk atoms used)
    out_dir : path
        Directory for x_fit.txt, coefs*.txt
    base_name : str
        Filename prefix (x_fit -> base_name_x_fit.txt)
    lam, sigma : float
        Kernel ridge params

    Returns
    -------
    X_fit, alphas, paths
    """
    ntrain = len(R_list)
    X_fit = np.array([compute_distance_matrix_upper(R, natmk) for R in R_list])
    # Build Y: (NKERNC*3, NTRAIN)
    abc_list = [_cpf_to_abc_flat(cpf) for cpf in charges_per_frame_list]
    nkernc3 = len(abc_list[0])
    Y_train = np.array(abc_list).T  # (NKERNC*3, NTRAIN)
    nkernc = nkernc3 // 3
    alphas = fit_kernel_ridge(X_fit, Y_train, lam=lam, sigma=sigma)
    paths = write_kernel_files(out_dir, X_fit, alphas, base_name=base_name)
    return X_fit, alphas, paths


def predict_charges_from_kernel(
    R: np.ndarray,
    X_fit: np.ndarray,
    alphas: np.ndarray,
    dq_per_charge: List[List[float]],
    natmk: int,
    sigma: float = 1.0,
) -> List:
    """
    Predict charges_per_frame from distance matrix using fitted kernel.

    Parameters
    ----------
    R : (n_atoms, 3)
    X_fit, alphas : from fit_kernel_from_training_data
    dq_per_charge : fixed DQ values (same structure as charges_per_frame)
    natmk : int
    sigma : float

    Returns
    -------
    charges_per_frame
    """
    x = compute_distance_matrix_upper(R, natmk).reshape(1, -1)
    pred = predict_kernel(x, X_fit, alphas, sigma=sigma)
    flat = pred.ravel()
    return _abc_flat_to_cpf(flat, dq_per_charge)

