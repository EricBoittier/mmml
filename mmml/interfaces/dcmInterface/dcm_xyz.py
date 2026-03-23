"""Generate dcm.xyz format (atoms + charge positions) from local frame coefficients."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from .convert import local_to_global
from .frame import compute_dcm_frame


def generate_dcm_xyz(
    R: np.ndarray,
    frames: List[Tuple[int, int, int]],
    charges_per_frame: List[List[Tuple[float, float, float, float]]],
    out_path: Union[str, Path],
) -> None:
    """
    Write dcm.xyz file matching CHARMM DCMXYZFILE format.

    Format:
        NATOMX + NALL
        dumped from DCM module
        C  x  y  z
        ...
        O  x  y  z  charge
        ...

    Parameters
    ----------
    R : np.ndarray
        Atom positions (n_atoms, 3)
    frames : list of (int, int, int)
        (atm1, atm2, atm3) 0-based per frame
    charges_per_frame : list of list of (AQ, BQ, CQ, DQ)
        charges for each frame's center atom
    out_path : path-like
        Output file path
    """
    R = np.asarray(R, dtype=float)
    n_atoms = R.shape[0]
    n_charges = sum(len(c) for c in charges_per_frame)
    total = n_atoms + n_charges

    lines = [str(total), "  dumped from DCM module"]
    for i in range(n_atoms):
        x, y, z = R[i]
        lines.append(f"C    {x:.4f}    {y:.4f}    {z:.4f}")

    for fr_idx, frame_atoms in enumerate(frames):
        frame_vectors = compute_dcm_frame(R, frame_atoms)
        center_atom = frame_atoms[0]
        X, Y, Z_vec = frame_vectors[0]
        atom_pos = R[center_atom]
        for aq, bq, cq, dq in charges_per_frame[fr_idx]:
            pos = local_to_global(atom_pos, aq, bq, cq, X, Y, Z_vec)
            lines.append(f"O    {pos[0]:.4f}    {pos[1]:.4f}    {pos[2]:.4f}    {dq:.4f}")

    Path(out_path).write_text("\n".join(lines) + "\n")
