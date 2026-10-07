"""Pipeline: DCMNet charge positions → CHARMM mdcm file."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .convert import global_to_local
from .frame import compute_dcm_frame
from .mdcm_writer import write_mdcm
from .topology import get_frames_meoh_like


def _compute_charges_per_frame(
    R: np.ndarray,
    charges: np.ndarray,
    positions: np.ndarray,
    frames: list[tuple[int, int, int]],
) -> list[list[tuple[float, float, float, float]]]:
    """Compute (AQ, BQ, CQ, DQ) for each frame from global positions."""
    n_charges = charges.shape[1]
    charges_per_frame: list[list[tuple[float, float, float, float]]] = []
    for frame_atoms in frames:
        frame_vectors = compute_dcm_frame(R, frame_atoms)
        center_atom = frame_atoms[0]
        X, Y, Z_vec = frame_vectors[0]
        atom_pos = R[center_atom]
        frame_charges = []
        for c in range(n_charges):
            aq, bq, cq = global_to_local(
                positions[center_atom, c], atom_pos, X, Y, Z_vec
            )
            frame_charges.append((aq, bq, cq, float(charges[center_atom, c])))
        charges_per_frame.append(frame_charges)
    return charges_per_frame


def dcmnet_to_mdcm(
    R: np.ndarray,
    Z: np.ndarray,
    dcmnet_charges: np.ndarray,
    dcmnet_charge_positions: np.ndarray,
    residue_name: str,
    out_path: Union[str, Path],
    frames: list[tuple[int, int, int]] | None = None,
) -> None:
    """
    Convert DCMNet outputs to CHARMM mdcm file.

    Parameters
    ----------
    R : np.ndarray
        Atom positions (n_atoms, 3)
    Z : np.ndarray
        Atomic numbers (n_atoms,)
    dcmnet_charges : np.ndarray
        (n_atoms, n_charges)
    dcmnet_charge_positions : np.ndarray
        (n_atoms, n_charges, 3) global positions
    residue_name : str
        Residue name for mdcm header
    out_path : path-like
        Output mdcm file path
    frames : list of (int, int, int), optional
        If None, derived from R and Z via get_frames_meoh_like
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=int)
    charges = np.asarray(dcmnet_charges, dtype=float)
    positions = np.asarray(dcmnet_charge_positions, dtype=float)

    n_atoms = R.shape[0]
    if charges.shape[0] != n_atoms or positions.shape[0] != n_atoms:
        raise ValueError("charges/positions first dim must match R")
    n_charges = charges.shape[1]
    if positions.shape[1] != n_charges or positions.shape[2] != 3:
        raise ValueError("positions must be (n_atoms, n_charges, 3)")

    if frames is None:
        frames = get_frames_meoh_like(R, Z)

    if len(frames) != n_atoms:
        raise ValueError(
            f"frames count {len(frames)} != n_atoms {n_atoms}; "
            "ensure one frame per atom"
        )

    charges_per_frame = _compute_charges_per_frame(R, charges, positions, frames)
    write_mdcm(out_path, residue_name, frames, charges_per_frame)


def build_mdcm_from_dcmnet(
    h5_path: Union[str, Path],
    frame_idx: int = 0,
    out_mdcm: Union[str, Path] = "meoh.mdcm",
    residue_name: str = "MEOH",
    average_over_frames: bool = False,
    frame_indices: Sequence[int] | None = None,
) -> tuple[list[tuple[int, int, int]], list[list[tuple[float, float, float, float]]]]:
    """
    Build mdcm from H5 file (charmm_ml_comparison format).

    Reads R, Z, dcmnet_charges, dcmnet_charge_positions and writes mdcm.

    Parameters
    ----------
    h5_path : path-like
        Path to H5 file
    frame_idx : int
        Single frame to use when average_over_frames=False
    out_mdcm : path-like
        Output mdcm file path
    residue_name : str
        Residue name for mdcm header
    average_over_frames : bool
        If True, average AQ, BQ, CQ, DQ over multiple conformations
    frame_indices : sequence of int, optional
        When average_over_frames=True, frames to average over.
        None = all frames in H5.

    Returns
    -------
    frames, charges_per_frame
        For use with generate_dcm_xyz if needed
    """
    import h5py

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with h5py.File(path, "r") as f:
        n_total = f["R"].shape[0]
        if "N" in f:
            n_atoms = int(f["N"][frame_idx])
        else:
            n_atoms = f["R"].shape[1]

    if average_over_frames:
        if frame_indices is None:
            frame_indices = list(range(n_total))
        else:
            frame_indices = list(frame_indices)
        if not frame_indices:
            raise ValueError("frame_indices must not be empty when average_over_frames=True")
        ref_idx = frame_indices[0]
        with h5py.File(path, "r") as f:
            n_atoms = int(f["N"][ref_idx]) if "N" in f else f["R"].shape[1]
        # Derive frames from first frame's geometry
        with h5py.File(path, "r") as f:
            R_ref = np.asarray(f["R"][ref_idx], dtype=float)[:n_atoms]
            Z = np.asarray(f["Z"][ref_idx], dtype=int)[:n_atoms]
        frames = get_frames_meoh_like(R_ref, Z)

        # Accumulate charges_per_frame across conformations
        acc: list[list[list[tuple[float, float, float, float]]]] = []
        # acc[conf][frame_idx][charge_idx] = (aq, bq, cq, dq)

        with h5py.File(path, "r") as f:
            has_n = "N" in f
            for idx in frame_indices:
                R = np.asarray(f["R"][idx], dtype=float)
                Z = np.asarray(f["Z"][idx], dtype=int)
                charges = np.asarray(f["dcmnet_charges"][idx], dtype=float)
                positions = np.asarray(f["dcmnet_charge_positions"][idx], dtype=float)
                if has_n:
                    n = int(f["N"][idx])
                    R, Z, charges, positions = (
                        R[:n],
                        Z[:n],
                        charges[:n],
                        positions[:n],
                    )
                cpf = _compute_charges_per_frame(R, charges, positions, frames)
                acc.append(cpf)

        # Average over conformations
        n_frames = len(frames)
        charges_per_frame = []
        for fr_idx in range(n_frames):
            n_q = len(acc[0][fr_idx])
            frame_avg = []
            for c in range(n_q):
                vals = np.array(
                    [acc[i][fr_idx][c] for i in range(len(acc))], dtype=float
                )
                avg = tuple(float(np.mean(vals[:, i])) for i in range(4))
                frame_avg.append(avg)
            charges_per_frame.append(frame_avg)
    else:
        with h5py.File(path, "r") as f:
            R = np.asarray(f["R"][frame_idx], dtype=float)
            Z = np.asarray(f["Z"][frame_idx], dtype=int)
            charges = np.asarray(f["dcmnet_charges"][frame_idx], dtype=float)
            positions = np.asarray(f["dcmnet_charge_positions"][frame_idx], dtype=float)
            n = int(f["N"][frame_idx]) if "N" in f else R.shape[0]

        R = R[:n]
        Z = Z[:n]
        charges = charges[:n]
        positions = positions[:n]
        frames = get_frames_meoh_like(R, Z)
        charges_per_frame = _compute_charges_per_frame(R, charges, positions, frames)

    write_mdcm(out_mdcm, residue_name, frames, charges_per_frame)
    return frames, charges_per_frame
