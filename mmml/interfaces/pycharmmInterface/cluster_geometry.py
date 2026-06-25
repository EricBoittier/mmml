"""Geometry helpers for notebooks, ASE calculators, and evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def reference_frame_geometry(
    path: str | Path,
    *,
    frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(atomic_numbers, positions)`` for one frame from an NPZ file.

    Accepts md-system handoff NPZs and QM reference trajectories (``R`` / ``Z`` / ``N``).
    Positions are in Angstrom; atomic numbers follow the NPZ frame order (use
    ``*_psf_order.npz`` when matching CHARMM PSF layout).
    """
    from mmml.cli.run.md_handoff import load_handoff_from_npz

    handoff = load_handoff_from_npz(Path(path).expanduser().resolve(), frame=frame)
    return (
        np.asarray(handoff.atomic_numbers, dtype=np.int32),
        np.asarray(handoff.positions, dtype=np.float64),
    )


def atoms_from_reference_npz(
    path: str | Path,
    *,
    frame: int = 0,
) -> Any:
    """Build an ASE ``Atoms`` object from a reference or handoff NPZ frame."""
    from ase import Atoms

    z, r = reference_frame_geometry(path, frame=frame)
    return Atoms(numbers=z, positions=r)
