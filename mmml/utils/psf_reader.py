"""Lightweight CHARMM PSF reader (atom types only, no PyCHARMM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PathLike = str | Path


def read_psf_atom_types(path: PathLike) -> np.ndarray:
    """Return per-atom MM types from a CHARMM PSF EXT ``!NATOM`` block.

    PSF atom line layout (EXT):
        index segid resid resname atomname type charge mass ...
    """
    psf_path = Path(path)
    lines = psf_path.read_text(encoding="utf-8", errors="replace").splitlines()
    natom: int | None = None
    start: int | None = None
    for idx, line in enumerate(lines):
        if "!NATOM" in line:
            parts = line.split()
            natom = int(parts[0])
            start = idx + 1
            break
    if natom is None or start is None:
        raise ValueError(f"Could not find !NATOM section in {psf_path}")

    types: list[str] = []
    for i in range(natom):
        line_idx = start + i
        if line_idx >= len(lines):
            raise ValueError(
                f"Expected {natom} atom lines in {psf_path}, found {len(types)} before EOF"
            )
        line = lines[line_idx]
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed PSF atom line {start + i + 1} in {psf_path}: {line!r}")
        types.append(str(parts[5]))

    if len(types) != natom:
        raise ValueError(f"Expected {natom} atom types, parsed {len(types)} in {psf_path}")
    return np.asarray(types, dtype=str)
