"""CHARMM DOMDEC NDIR helpers for MPI rank counts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ALL c47 builds (including MMML native) enforce this in domdec.F90:
# each NDIR axis must be 1 or >= 8 (values 2–7 are rejected at runtime).
# Confirmed by runtime error: "x-direction must have (a) 1 node or (b) at least 8 nodes"
_MIN_AXIS_NODES = 8

# Vendored CHARMM doc (extract setup/charmm via setup/install.sh or charmm.tar.xz).
CHARMM_DOMDEC_DOC = "setup/charmm/doc/domdec.info"
CHARMM_ENERGY_DOC = "setup/charmm/doc/energy.info"

_DOMDEC_NP_HINT = (
    "CHARMM c47 domdec.F90 requires each NDIR axis to be 1 or >=8. "
    "Minimum useful MPI count is 8 (NDIR 8 1 1) with a >=152 Å box. "
    "For liquid-density MLPot scaling use MMML spatial MPI instead of DOMDEC."
)


def _axis_valid_c47(n: int) -> bool:
    return n == 1 or n >= _MIN_AXIS_NODES


def min_domdec_mpi_ranks(*, allow_serial: bool = True, strict_c47_axis_rule: bool = True) -> int:
    """Smallest ``np>1`` compatible with the active NDIR policy.

    ``strict_c47_axis_rule`` defaults to ``True`` because ALL c47 builds enforce
    the "1 or >=8 per axis" constraint in ``domdec.F90``.
    """
    if allow_serial:
        return 1
    return _MIN_AXIS_NODES if strict_c47_axis_rule else 2


def suggest_domdec_ndir(n_ranks: int, *, strict_c47_axis_rule: bool = True) -> tuple[int, int, int]:
    """Return ``(nx, ny, nz)`` for ``energy domdec ndir nx ny nz``.

    ``strict_c47_axis_rule`` defaults to ``True`` because ALL c47 builds (including
    MMML native) enforce the "1 or >=8 per axis" rule in ``domdec.F90``.
    Pass ``strict_c47_axis_rule=False`` only for non-c47 custom builds.
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
        hint = _DOMDEC_NP_HINT if strict_c47_axis_rule else f"factor n_ranks={n} into NDIR nx*ny*nz"
        raise ValueError(f"no valid DOMDEC NDIR for n_ranks={n}: {hint}")

    candidates.sort()
    nx, ny, nz = candidates[0][4], candidates[0][5], candidates[0][6]
    return (nx, ny, nz)


def format_domdec_ndir(n_ranks: int, *, strict_c47_axis_rule: bool = True) -> str:
    nx, ny, nz = suggest_domdec_ndir(n_ranks, strict_c47_axis_rule=strict_c47_axis_rule)
    return f"{nx} {ny} {nz}"


def format_domdec_charmm_commands(n_ranks: int, *, strict_c47_axis_rule: bool = True) -> str:
    """Alias for :func:`format_domdec_tier3_energy_block` (single continued ENERGY line)."""
    return format_domdec_tier3_energy_block(n_ranks, strict_c47_axis_rule=strict_c47_axis_rule)


def format_domdec_tier3_energy_block(
    n_ranks: int,
    *,
    cutnb: float = 15.0,
    ctonnb: float = 10.83,
    ctofnb: float = 14.17,
    cutim: float = 15.0,
    ctexnb: float = 15.0,
    strict_c47_axis_rule: bool = True,
) -> str:
    """Continued ENERGY command with nonbond keywords and ``domd ndir`` on the same command.

    CHARMM ``GETE0`` (``source/energy/eutil.F90``) calls ``indxa(..., 'DOMD')``; the energy
    line must include the 4-char token ``domd`` (``domdec.info`` Example 1 uses the word
    ``domdec``, which ``indxa`` matches as ``DOMD`` when it is its own token on the line).
    See ``setup/charmm/doc/domdec.info`` and ``setup/charmm/doc/energy.info`` [ domdec-spec ].
    """
    ndir = format_domdec_ndir(n_ranks, strict_c47_axis_rule=strict_c47_axis_rule)
    # GETE0 (eutil.F90) attaches domdec via indxa(..., 'DOMD') — use domd not domdec.
    keyword = (os.environ.get("DOMDEC_ENERGY_KEYWORD") or "domd").strip().lower()
    split = " split off" if n_ranks > 1 else ""
    return (
        f"energy cutnb {cutnb} cutim {cutim} ctonnb {ctonnb} ctofnb {ctofnb} -\n"
        f"  vfswitch vatom cdie eps 1.0 -\n"
        f"  {keyword} ndir {ndir}{split}"
    )


def min_domdec_crystal_side_A(
    n_ranks: int,
    cutnb: float = 15.0,
    group_halo: float = 4.0,
    *,
    strict_c47_axis_rule: bool = True,
) -> float:
    """Minimum cubic lattice side (Å) for DOMDEC to satisfy the cutoff geometry.

    CHARMM's DOMDEC geometry constraint for an axis split into N domains:

        L ≥ 2 · RCUT · N / (N − 1)

    where RCUT = cutnb + max_group_radius (``group_halo``).

    This ensures that the minimum-image distance across any domain boundary
    (``L · (N−1)/N``) is at least ``2·RCUT``, so no two atoms whose images
    straddle the same domain boundary are double-counted or missed.

    For N=1 the axis is not decomposed; the box need only accommodate the
    nonbond list (``L > 2·RCUT``).

    Examples with RCUT = 15 + 4 = 19 Å:
      N=2  (np=2,  non-strict): L ≥ 2·19·2/1  = 76.0 Å
      N=8  (np=8,  strict    ): L ≥ 2·19·8/7  ≈ 43.4 Å
      N=16 (np=16, strict    ): L ≥ 2·19·16/15 ≈ 40.5 Å
    """
    n = int(n_ranks)
    rcut = float(cutnb) + float(group_halo)
    if n <= 1:
        return 2.0 * rcut
    nx, ny, nz = suggest_domdec_ndir(n, strict_c47_axis_rule=strict_c47_axis_rule)
    min_sides = []
    for n_ax in (nx, ny, nz):
        if n_ax <= 1:
            continue
        min_sides.append(2.0 * rcut * n_ax / (n_ax - 1))
    return max(min_sides) if min_sides else 2.0 * rcut


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
