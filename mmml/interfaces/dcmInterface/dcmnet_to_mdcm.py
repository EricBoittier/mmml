"""Pipeline: DCMNet charge positions → CHARMM mdcm file."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .convert import global_to_local
from .frame import compute_dcm_frame
from .mdcm_writer import write_mdcm
from .topology import get_frames_meoh_like


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

    charges_per_frame: list[list[tuple[float, float, float, float]]] = []
    for fr_idx, frame_atoms in enumerate(frames):
        frame_vectors = compute_dcm_frame(R, frame_atoms)
        # Charges for this frame are on the center atom (position 0)
        center_atom = frame_atoms[0]
        X, Y, Z_vec = frame_vectors[0]
        atom_pos = R[center_atom]
        frame_charges = []
        for c in range(n_charges):
            global_pos = positions[center_atom, c]
            aq, bq, cq = global_to_local(global_pos, atom_pos, X, Y, Z_vec)
            dq = float(charges[center_atom, c])
            frame_charges.append((aq, bq, cq, dq))
        charges_per_frame.append(frame_charges)

    write_mdcm(out_path, residue_name, frames, charges_per_frame)


def build_mdcm_from_dcmnet(
    h5_path: Union[str, Path],
    frame_idx: int,
    out_mdcm: Union[str, Path],
    residue_name: str = "MEOH",
) -> tuple[list[tuple[int, int, int]], list[list[tuple[float, float, float, float]]]]:
    """
    Build mdcm from H5 file (charmm_ml_comparison format).

    Reads R, Z, dcmnet_charges, dcmnet_charge_positions and writes mdcm.

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
        R = np.asarray(f["R"][frame_idx], dtype=float)
        Z = np.asarray(f["Z"][frame_idx], dtype=int)
        charges = np.asarray(f["dcmnet_charges"][frame_idx], dtype=float)
        positions = np.asarray(f["dcmnet_charge_positions"][frame_idx], dtype=float)
        n = int(f["N"][frame_idx]) if "N" in f else R.shape[0]

    # Clip to actual atoms if padded
    R = R[:n]
    Z = Z[:n]
    charges = charges[:n]
    positions = positions[:n]

    frames = get_frames_meoh_like(R, Z)
    dcmnet_to_mdcm(R, Z, charges, positions, residue_name, out_mdcm, frames=frames)

    # Rebuild charges_per_frame for caller
    charges_per_frame = []
    for fr_idx, frame_atoms in enumerate(frames):
        frame_vectors = compute_dcm_frame(R, frame_atoms)
        center_atom = frame_atoms[0]
        X, Y, Z_vec = frame_vectors[0]
        atom_pos = R[center_atom]
        n_charges = charges.shape[1]
        frame_charges = []
        for c in range(n_charges):
            aq, bq, cq = global_to_local(
                positions[center_atom, c], atom_pos, X, Y, Z_vec
            )
            frame_charges.append((aq, bq, cq, float(charges[center_atom, c])))
        charges_per_frame.append(frame_charges)

    return frames, charges_per_frame
