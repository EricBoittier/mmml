"""
Utility functions for run_sim and related runners.

Extracted from run_sim.py to reduce duplication and improve clarity.
"""
from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np


def normalize_n_atoms_monomer(
    raw_n_atoms_monomer: Union[int, List[int], tuple, np.ndarray],
    n_monomers: int,
) -> Tuple[List[int], int, int, int, np.ndarray]:
    """Normalize n_atoms_monomer (int or list) to atoms_per_monomer_list and offsets.

    Args:
        raw_n_atoms_monomer: Either an int (uniform) or sequence of per-monomer counts.
        n_monomers: Number of monomers (used when raw is int).

    Returns:
        (atoms_per_monomer_list, n_monomers, total_atoms, n_atoms_first, monomer_offsets)
    """
    if isinstance(raw_n_atoms_monomer, (list, tuple, np.ndarray)):
        atoms_per_monomer_list = [int(x) for x in raw_n_atoms_monomer]
        n_monomers = len(atoms_per_monomer_list)
        total_atoms = sum(atoms_per_monomer_list)
        n_atoms_first = atoms_per_monomer_list[0]
    else:
        n_atoms_first = int(raw_n_atoms_monomer)
        atoms_per_monomer_list = [n_atoms_first] * n_monomers
        total_atoms = n_atoms_first * n_monomers

    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    for _mi, _na in enumerate(atoms_per_monomer_list):
        monomer_offsets[_mi + 1] = monomer_offsets[_mi] + _na

    return atoms_per_monomer_list, n_monomers, total_atoms, n_atoms_first, monomer_offsets


def get_steps_per_frame(args: Any) -> int:
    """Get steps per recorded frame from args.

    Uses steps_per_recording if set, else 25 for NPT with cell, 1000 otherwise.
    """
    steps = getattr(args, "steps_per_recording", None)
    if steps is not None:
        return steps
    if args.ensemble == "npt" and args.cell is not None:
        return 25
    return 1000
