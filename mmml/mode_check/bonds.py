"""Bond-pair inference for monomer / small-cluster mode checks."""

from __future__ import annotations

import numpy as np

# Generous covalent cutoffs (Å) for X–H pairing inside a monomer.
_XH_CUTOFF = {
    6: 1.30,  # C–H
    7: 1.25,  # N–H
    8: 1.30,  # O–H
    9: 1.20,  # F–H
    16: 1.50,  # S–H
    17: 1.45,  # Cl–H
}


def monomer_slices(atoms_per_monomer: list[int] | tuple[int, ...]) -> list[slice]:
    """Return index slices for each monomer from a per-monomer atom-count list."""
    offsets = np.cumsum([0, *[int(n) for n in atoms_per_monomer]])
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(atoms_per_monomer))]


def infer_xh_bond_pairs(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    *,
    atoms_per_monomer: list[int] | tuple[int, ...] | None = None,
    max_distance: float | None = None,
) -> list[tuple[int, int]]:
    """Infer heavy–hydrogen pairs within each monomer (or the whole system).

    For each H, choose the nearest heavy atom (Z>1) inside the same monomer
    subject to a covalent cutoff. Returns ``(heavy, H)`` index pairs.
    """
    z = np.asarray(atomic_numbers, dtype=int)
    pos = np.asarray(positions, dtype=float)
    if z.shape[0] != pos.shape[0]:
        raise ValueError("atomic_numbers and positions length mismatch")
    n = int(z.shape[0])
    if atoms_per_monomer is None:
        slices = [slice(0, n)]
    else:
        if int(sum(atoms_per_monomer)) != n:
            raise ValueError(
                f"atoms_per_monomer sum ({sum(atoms_per_monomer)}) != natoms ({n})"
            )
        slices = monomer_slices(atoms_per_monomer)

    pairs: list[tuple[int, int]] = []
    for sl in slices:
        idx = np.arange(sl.start, sl.stop, dtype=int)
        heavies = idx[z[idx] > 1]
        hydrogens = idx[z[idx] == 1]
        for h in hydrogens:
            if heavies.size == 0:
                continue
            d = np.linalg.norm(pos[heavies] - pos[h], axis=1)
            j = int(np.argmin(d))
            heavy = int(heavies[j])
            r = float(d[j])
            cutoff = float(max_distance) if max_distance is not None else float(
                _XH_CUTOFF.get(int(z[heavy]), 1.40)
            )
            if r <= cutoff:
                pairs.append((heavy, int(h)))
    return pairs


def tip3_oh_pairs(n_molecules: int) -> list[tuple[int, int]]:
    """Canonical TIP3 layout: O,H,H per molecule → two OH bonds each."""
    n = int(n_molecules)
    if n < 1:
        raise ValueError("n_molecules must be >= 1")
    pairs: list[tuple[int, int]] = []
    for m in range(n):
        o = 3 * m
        pairs.append((o, o + 1))
        pairs.append((o, o + 2))
    return pairs
