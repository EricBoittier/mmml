"""CHARMM ABNR with LATTice for CRYSTAL box optimization during mini."""

from __future__ import annotations

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    get_charmm_positions_array,
    sync_charmm_positions,
)


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


def run_charmm_lattice_abnr(
    *,
    nstep: int,
    tolenr: float,
    tolgrd: float,
    nocoords: bool = False,
    verbose: bool = True,
    nbxmod: int = 5,
) -> float | None:
    """Run CHARMM ``MINI ABNR LATTice`` to optimize the unit cell (and optionally coords).

    PyCHARMM's :func:`pycharmm.minimize.run_abnr` C binding does not pass the SD
    ``lattice`` flag, so this uses a CHARMM script command instead.
    """
    if int(nstep) <= 0:
        return None

    import pycharmm.script

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        apply_pbc_nbonds,
        resolve_charmm_cubic_box_side_A,
    )
    kwargs: dict[str, object] = {
        "lattice": True,
        "nstep": int(nstep),
        "tolenr": float(tolenr),
        "tolgrd": float(tolgrd),
    }
    if nocoords:
        kwargs["nocoords"] = True
    if verbose:
        mode = "box only" if nocoords else "coords + box"
        print(
            f"CHARMM lattice ABNR: nstep={int(nstep)} ({mode})",
            flush=True,
        )
    with charmm_quiet_output():
        pycharmm.script.CommandScript("mini abnr", **kwargs).run()
    side, source = resolve_charmm_cubic_box_side_A(fallback_side_A=None)
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
    )
    sync_charmm_positions(get_charmm_positions_array())
    if new_side is not None:
        return float(new_side)
    return box_side
