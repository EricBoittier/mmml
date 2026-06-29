"""CHARMM DOMDEC NDIR helpers for MPI rank counts."""

from __future__ import annotations

import json
import re
from pathlib import Path

# c47 site builds: each NDIR axis must be 1 or >= 8 (2–7 nodes forbidden).
_MIN_AXIS_NODES = 8


def _axis_valid(n: int) -> bool:
    return n == 1 or n >= _MIN_AXIS_NODES


def min_domdec_mpi_ranks(*, allow_serial: bool = True) -> int:
    """Smallest ``np>1`` compatible with c47-style per-axis NDIR limits."""
    if allow_serial:
        return 1
    return _MIN_AXIS_NODES


def suggest_domdec_ndir(n_ranks: int) -> tuple[int, int, int]:
    """Return ``(nx, ny, nz)`` for ``ENERGY DOMDEC NDIR nx ny nz``.

    c47 site builds reject 2–7 nodes on **any** axis (each axis must be 1 or >= 8).
    Prefer a 1-D decomposition (e.g. ``8 1 1`` for ``np=8``).
    """
    n = int(n_ranks)
    if n <= 1:
        return (1, 1, 1)

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for nx in range(1, n + 1):
        if n % nx:
            continue
        rest = n // nx
        for ny in range(1, rest + 1):
            if rest % ny:
                continue
            nz = rest // ny
            if not (_axis_valid(nx) and _axis_valid(ny) and _axis_valid(nz)):
                continue
            score = (
                0 if ny == 1 and nz == 1 else 1,
                0 if nx == n else 1,
                0 if nx >= nz else 1,
                -nx,
            )
            candidates.append((*score, nx, ny, nz))

    if not candidates:
        raise ValueError(
            f"no valid DOMDEC NDIR for n_ranks={n}: c47 requires each axis be 1 or >= {_MIN_AXIS_NODES} "
            f"(np=2..7 cannot work; use MMML_MPI_NP=8 or a newer CHARMM build)"
        )

    candidates.sort()
    nx, ny, nz = candidates[0][4], candidates[0][5], candidates[0][6]
    return (nx, ny, nz)


def format_domdec_ndir(n_ranks: int) -> str:
    nx, ny, nz = suggest_domdec_ndir(n_ranks)
    return f"{nx} {ny} {nz}"


def min_domdec_crystal_side_A(
    n_ranks: int,
    cutnb: float = 15.0,
    group_halo: float = 4.0,
) -> float:
    """Minimum cubic lattice side (Å) for DOMDEC with the chosen NDIR."""
    n = int(n_ranks)
    per_domain = float(cutnb) + float(group_halo)
    if n <= 1:
        return per_domain
    nx, ny, nz = suggest_domdec_ndir(n)
    n_split = max(nx, ny, nz)
    return float(n_split * per_domain)


def _read_prep_box_side_A(box_dir: Path) -> float | None:
    box_json = box_dir / "box.json"
    if box_json.is_file():
        side = json.loads(box_json.read_text()).get("box_side_A")
        if side is not None:
            return float(side)
    match = re.search(r"_l([0-9]+)$", box_dir.name)
    if match:
        return float(match.group(1))
    return None


def pick_domdec_prep_dir(
    boxes_root: Path,
    *,
    n_dcm: int,
    min_side_A: float = 0.0,
) -> Path | None:
    """Pick prep dir: smallest side >= min_side_A, else newest when min_side_A=0."""
    pattern = f"domdec_dcm{n_dcm}_l"
    candidates: list[tuple[float, float, Path]] = []
    if not boxes_root.is_dir():
        return None
    for entry in boxes_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(pattern):
            continue
        psf = entry / "model.psf"
        if not psf.is_file():
            continue
        side = _read_prep_box_side_A(entry)
        if side is None:
            continue
        if side + 1e-6 < float(min_side_A):
            continue
        candidates.append((side, psf.stat().st_mtime, entry))
    if not candidates:
        return None
    if min_side_A > 0:
        candidates.sort(key=lambda item: (item[0], -item[1]))
    else:
        candidates.sort(key=lambda item: -item[1])
    return candidates[0][2]
