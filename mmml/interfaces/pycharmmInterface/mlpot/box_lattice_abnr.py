"""CHARMM ABNR with LATTice for CRYSTAL box optimization during mini."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    get_charmm_positions_array,
    sync_charmm_positions,
)

PathLike = Union[str, Path]


def should_run_mini_lattice_abnr(
    args: argparse.Namespace,
    *,
    charmm_pbc: bool,
    stages: list[str],
) -> bool:
    """True when lattice ABNR should run after coordinate-only CHARMM MM mini."""
    nstep = int(getattr(args, "mini_lattice_abnr_steps", 0) or 0)
    if nstep <= 0 or not charmm_pbc:
        return False
    if "mini" not in stages:
        return False
    if getattr(args, "box_size", None) is not None and not bool(
        getattr(args, "mini_lattice_abnr_allow_fixed_box", False)
    ):
        return False
    return True


def _run_lattice_minimize_c_api(
    *,
    nstep: int,
    tolenr: float,
    tolgrd: float,
    nocoords: bool,
) -> None:
    """Lattice box optimization via KEY_LIBRARY minimize C API (no ``mini`` script)."""
    import pycharmm.minimize as charm_min

    kwargs: dict[str, object] = {
        "lattice": True,
        "nstep": int(nstep),
        "tolenr": float(tolenr),
        "tolgrd": float(tolgrd),
    }
    if nocoords:
        kwargs["nocoords"] = True
    if not charm_min.run_abnr(**kwargs):
        raise RuntimeError("CHARMM lattice minimize (ABNR/SD) failed")


def run_charmm_lattice_abnr(
    *,
    nstep: int,
    tolenr: float,
    tolgrd: float,
    nocoords: bool = False,
    verbose: bool = True,
    nbxmod: int = 5,
    fallback_side_A: float | None = None,
    restart_path: PathLike | None = None,
    allow_prepare_pbc: bool = True,
) -> float | None:
    """Run CHARMM lattice minimization to optimize the unit cell (and optionally coords).

    Uses :func:`pycharmm.minimize.run_abnr` with ``lattice=True`` (C API). When
    ``minimize_run_abnr_lattice`` is absent from ``libcharmm``, falls back to SD
    with the same lattice flags.

    After MM pretreat, ``pbound_get_size`` often reads zero in Python while IMAGE
    lists remain valid in Fortran. Pass ``fallback_side_A`` / ``restart_path`` so
    post-minimize box resolution does not fail when live pbound is inactive.
    """
    if int(nstep) <= 0:
        return None

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        charmm_crystal_is_active,
        probe_charmm_cubic_box_side_A,
        reinstall_charmm_crystal_for_lattice_abnr,
        resolve_charmm_cubic_box_side_A,
    )
    restore_side = fallback_side_A
    if restore_side is None or float(restore_side) <= 0.0:
        probed, _ = probe_charmm_cubic_box_side_A()
        restore_side = probed
    if restore_side is not None and float(restore_side) > 0.0:
        from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
            charmm_crystal_abnr_ready,
        )

        if not charmm_crystal_abnr_ready(float(restore_side)):
            reinstall_charmm_crystal_for_lattice_abnr(
                float(restore_side),
                quiet=not verbose,
                allow_prepare_pbc=bool(allow_prepare_pbc),
            )
    if verbose:
        mode = "box only" if nocoords else "coords + box"
        print(
            f"CHARMM lattice ABNR: nstep={int(nstep)} ({mode})",
            flush=True,
        )
    with charmm_quiet_output():
        _run_lattice_minimize_c_api(
            nstep=int(nstep),
            tolenr=float(tolenr),
            tolgrd=float(tolgrd),
            nocoords=nocoords,
        )
    restart_for_resolve = restart_path
    if restart_for_resolve is not None and charmm_crystal_is_active():
        restart_for_resolve = None
    side, source = resolve_charmm_cubic_box_side_A(
        fallback_side_A=fallback_side_A,
        restart_path=restart_for_resolve,
    )
    if side is not None and float(side) > 0.0:
        apply_pbc_nbonds(nbxmod=int(nbxmod), cubic_box_side_A=float(side))
        if verbose:
            print(
                f"CHARMM lattice ABNR end: cubic L={float(side):.3f} Å (source={source})",
                flush=True,
            )
        return float(side)
    return None


def run_mini_lattice_abnr(
    args: argparse.Namespace,
    *,
    box_side: float | None,
    use_pbc: bool,
    pretreat_restart: PathLike | None = None,
) -> float | None:
    """Lattice ABNR leg between coordinate CHARMM MM mini and MLpot registration."""
    if not use_pbc:
        raise ValueError("mini lattice ABNR requires PBC")
    nstep = int(getattr(args, "mini_lattice_abnr_steps", 0) or 0)
    if nstep <= 0:
        return box_side
    tolenr = float(getattr(args, "charmm_tolenr", 1e-3))
    tolgrd = float(getattr(args, "charmm_tolgrd", 1e-3))
    nocoords = bool(getattr(args, "mini_lattice_abnr_nocoords", False))
    if not args.quiet:
        print(
            f"\nMini lattice ABNR: optimizing unit cell before MLpot SD",
            flush=True,
        )
    new_side = run_charmm_lattice_abnr(
        nstep=nstep,
        tolenr=tolenr,
        tolgrd=tolgrd,
        nocoords=nocoords,
        verbose=not bool(args.quiet),
        fallback_side_A=box_side,
        restart_path=pretreat_restart,
    )
    sync_charmm_positions(get_charmm_positions_array())
    if new_side is not None:
        return float(new_side)
    return box_side
