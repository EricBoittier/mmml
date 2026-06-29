"""CHARMM DOMDEC NDIR helpers for MPI rank counts."""

from __future__ import annotations

import json
import re
from pathlib import Path

# NIH site c47 builds: each NDIR axis must be 1 or >= 8 (2–7 nodes forbidden).
_MIN_AXIS_NODES = 8


def _axis_valid_c47(n: int) -> bool:
    return n == 1 or n >= _MIN_AXIS_NODES


def min_domdec_mpi_ranks(*, allow_serial: bool = True, strict_c47_axis_rule: bool = False) -> int:
    """Smallest ``np>1`` compatible with the active NDIR policy."""
    if allow_serial:
        return 1
    return _MIN_AXIS_NODES if strict_c47_axis_rule else 2


def suggest_domdec_ndir(n_ranks: int, *, strict_c47_axis_rule: bool = False) -> tuple[int, int, int]:
    """Return ``(nx, ny, nz)`` for ``domdec on ndir nx ny nz``.

    Default (MMML tier3): allow ``NDIR 2 1 1`` etc. for small dense boxes at ``np=2``.

    With ``strict_c47_axis_rule=True`` (site c47): each axis must be 1 or >= 8.
    """
    n = int(n_ranks)
    if n <= 1:
        return (1, 1, 1)

    def axis_ok(v: int) -> bool:
        return _axis_valid_c47(v) if strict_c47_axis_rule else v >= 1

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for nx in range(1, n + 1):
        if n % nx:
            continue
        rest = n // nx
        for ny in range(1, rest + 1):
            if rest % ny:
                continue
            nz = rest // ny
            if not (axis_ok(nx) and axis_ok(ny) and axis_ok(nz)):
                continue
            score = (
                0 if ny == 1 and nz == 1 else 1,
                0 if nx == n else 1,
                0 if nx >= nz else 1,
                -nx,
            )
            candidates.append((*score, nx, ny, nz))

    if not candidates:
        hint = (
            f"use MMML_MPI_NP=8 with strict_c47_axis_rule, or MMML native CHARMM at np=2"
            if strict_c47_axis_rule
            else f"factor n_ranks={n} into NDIR nx*ny*nz"
        )
        raise ValueError(f"no valid DOMDEC NDIR for n_ranks={n}: {hint}")

    candidates.sort()
    nx, ny, nz = candidates[0][4], candidates[0][5], candidates[0][6]
    return (nx, ny, nz)


def format_domdec_ndir(n_ranks: int, *, strict_c47_axis_rule: bool = False) -> str:
    nx, ny, nz = suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=strict_c47_axis_rule)
    return f"{nx} {ny} {nz}"


def format_domdec_charmm_commands(n_ranks: int, *, strict_c47_axis_rule: bool = False) -> str:
    """CHARMM stream block: ``domdec on ndir …`` then ``energy`` (not ``energy domdec …``)."""
    ndir = format_domdec_ndir(n_ranks, strict_c47_axis_rule=strict_c47_axis_rule)
    return f"domdec on ndir {ndir}\nenergy"


def min_domdec_crystal_side_A(
    n_ranks: int,
    cutnb: float = 15.0,
    group_halo: float = 4.0,
    *,
    strict_c47_axis_rule: bool = False,
) -> float:
    """Minimum cubic lattice side (Å) so each DOMDEC subdomain fits the cutoff."""
    n = int(n_ranks)
    per_domain = float(cutnb) + float(group_halo)
    if n <= 1:
        return per_domain
    nx, ny, nz = suggest_domdec_ndir(n, strict_c47_axis_rule=strict_c47_axis_rule)
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
    prefer_smallest: bool = True,
) -> Path | None:
    """Pick a prep dir with ``box_side_A >= min_side_A``.

    For tier3, ``prefer_smallest=True`` keeps dense ~40 Å preps (PBC images) over huge dilute boxes.
    """
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
    if prefer_smallest or min_side_A > 0:
        candidates.sort(key=lambda item: (item[0], -item[1]))
    else:
        candidates.sort(key=lambda item: -item[1])
    return candidates[0][2]
