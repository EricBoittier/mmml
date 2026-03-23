"""Write CHARMM mdcm DCM input files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union


def write_mdcm(
    path: Union[str, Path],
    residue_name: str,
    frames: List[Tuple[int, int, int]],
    charges_per_frame: List[List[Tuple[float, float, float, float]]],
) -> None:
    """
    Write mdcm file for CHARMM DCM module.

    Each frame assigns charges to the first atom only (the frame center).
    Format: one frame per (atm1, atm2, atm3); charges are (AQ, BQ, CQ, DQ).

    Parameters
    ----------
    path : path-like
        Output file path
    residue_name : str
        Residue name (e.g. MEOH)
    frames : list of (int, int, int)
        Each (atm1, atm2, atm3) 0-based. CHARMM expects 1-based in file.
    charges_per_frame : list of list of (AQ, BQ, CQ, DQ)
        charges_per_frame[i] = charges for frame i, center atom only
    """
    path = Path(path)
    n_frames = len(frames)
    if len(charges_per_frame) != n_frames:
        raise ValueError(
            f"charges_per_frame length {len(charges_per_frame)} != frames {n_frames}"
        )

    lines = [
        "1 0",
        residue_name,
        str(n_frames),
    ]
    for fr_idx, (a1, a2, a3) in enumerate(frames):
        # 1-indexed for CHARMM
        lines.append(f"{a1 + 1} {a2 + 1} {a3 + 1} BO")
        charges = charges_per_frame[fr_idx]
        nq = len(charges)
        lines.append(f"{nq} 0")  # n_charges, n_polarizabilities for atom 1
        for aq, bq, cq, dq in charges:
            lines.append(f"{aq:.6f} {bq:.6f} {cq:.6f} {dq:.6f}")
        # Atom 2 and 3: no charges
        lines.append("0 0")
        lines.append("0 0")
    path.write_text("\n".join(lines) + "\n")
