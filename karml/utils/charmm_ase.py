"""CHARMM PSF → ASE helpers for structure export and plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def element_symbols_from_psf(psf_path: Path | str, *, n_atoms: int) -> list[str]:
    """Heavy-atom element letter + H for hydrogens (PSF atom-name heuristic)."""
    path = Path(psf_path)
    symbols: list[str] = []
    in_atoms = False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip().startswith("*"):
                in_atoms = False
                continue
            if "!NATOM" in line:
                in_atoms = True
                continue
            if not in_atoms:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            name = parts[3]
            if name.startswith("H"):
                symbols.append("H")
            elif name.startswith("O"):
                symbols.append("O")
            elif name.startswith("N"):
                symbols.append("N")
            elif name.startswith("S"):
                symbols.append("S")
            else:
                symbols.append("C")
            if len(symbols) >= n_atoms:
                break
    if len(symbols) != n_atoms:
        raise RuntimeError(
            f"PSF parse got {len(symbols)} symbols, expected {n_atoms} from {path}"
        )
    return symbols


def atoms_from_psf_box(
    psf_path: Path | str,
    positions: np.ndarray,
    *,
    box_side_A: float | None = None,
    pbc: bool = True,
):
    """Build periodic ASE ``Atoms`` from a CHARMM PSF and coordinate array."""
    from ase import Atoms

    pos = np.asarray(positions, dtype=float)
    n_atoms = int(pos.shape[0])
    symbols = element_symbols_from_psf(psf_path, n_atoms=n_atoms)
    cell = None
    pbc_flags = False
    if box_side_A is not None and float(box_side_A) > 0:
        side = float(box_side_A)
        cell = np.diag([side, side, side])
        pbc_flags = bool(pbc)
    return Atoms(symbols=symbols, positions=pos, cell=cell, pbc=pbc_flags)


__all__ = ["atoms_from_psf_box", "element_symbols_from_psf"]
