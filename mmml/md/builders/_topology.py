"""Molecule partitioning from bonded connectivity (builder-side helpers)."""

from __future__ import annotations

import numpy as np

__all__ = ["molecule_ids_from_bonds", "monomer_indices_from_mol_id"]


def molecule_ids_from_bonds(n_atoms: int, bonds: np.ndarray) -> np.ndarray:
    """Assign a contiguous molecule id per atom via connected components of bonds.

    ``bonds`` is an ``(M, 2)`` array of **0-based** atom indices (as
    ``parse_psf_ext`` returns). Atoms with no bonds are their own molecule. Ids
    are renumbered 0..K-1 in order of first appearance so they are stable and gap-free.
    """
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    for i, j in bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[rj] = ri

    # renumber roots to contiguous ids by first appearance
    remap: dict[int, int] = {}
    mol_id = np.empty(n_atoms, dtype=np.int32)
    for atom in range(n_atoms):
        root = find(atom)
        if root not in remap:
            remap[root] = len(remap)
        mol_id[atom] = remap[root]
    return mol_id


def monomer_indices_from_mol_id(mol_id: np.ndarray) -> list[np.ndarray]:
    """Group atom indices by molecule id into a list of ``(n_i,)`` index arrays."""
    mol_id = np.asarray(mol_id)
    return [np.where(mol_id == m)[0] for m in range(int(mol_id.max()) + 1)] if mol_id.size else []
