"""CHARMM-free ACO cluster geometry for CPU MD examples."""

from __future__ import annotations

import numpy as np
from ase.io import read as ase_read

from mmml.paths import default_aco_template_pdb

ACO_ATOMS_PER_MONOMER = 10


def aco_dimer_cluster(
    *,
    n_monomers: int = 2,
    spacing: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Z, positions) for ACO×N from bundled PDB templates (no PyCHARMM)."""
    if n_monomers < 1:
        raise ValueError(f"n_monomers must be >= 1, got {n_monomers}")

    monomer = ase_read(str(default_aco_template_pdb()))
    z_mono = np.asarray(monomer.get_atomic_numbers(), dtype=np.int32)
    r_mono = np.asarray(monomer.get_positions(), dtype=np.float64)
    if z_mono.shape[0] != ACO_ATOMS_PER_MONOMER:
        raise RuntimeError(
            f"expected {ACO_ATOMS_PER_MONOMER} atoms in ACO template, got {z_mono.shape[0]}"
        )

    n_side = int(np.ceil(np.sqrt(n_monomers)))
    chunks_z: list[np.ndarray] = []
    chunks_r: list[np.ndarray] = []
    for mi in range(n_monomers):
        r_i = r_mono.copy()
        r_i -= r_i.mean(axis=0)
        shift = np.array(
            [(mi % n_side) * spacing, (mi // n_side) * spacing, 0.0],
            dtype=np.float64,
        )
        r_i += shift
        chunks_z.append(z_mono)
        chunks_r.append(r_i)

    z = np.concatenate(chunks_z)
    r = np.vstack(chunks_r)
    expected = ACO_ATOMS_PER_MONOMER * n_monomers
    if z.shape[0] != expected:
        raise RuntimeError(f"expected {expected} atoms, got {z.shape[0]}")
    return z, r
