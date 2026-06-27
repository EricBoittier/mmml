"""Short CPT NPT box equilibration during the staged mini step."""

from __future__ import annotations

from pathlib import Path

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array, sync_charmm_positions


def configure_liquid_box_mini_equil_args(
    args: argparse.Namespace,
    *,
    box_side_A: float,
) -> None:
    """Pin the certified cubic box and use Hoover NVT (not CPT NPT) for mini equil.

    ``liquid-box`` already sizes the cell from MC density / target ρ. CPT barostat
    legs on that geometry routinely spike pressure and abort early (echeck / step 240).
    """
    side = float(box_side_A)
    if side <= 0.0:
        raise ValueError(f"box_side_A must be > 0, got {side}")
    args.box_size = side
    args.mini_box_equil_allow_fixed_box = True
    args.mini_box_equil_fixed_nvt = True


def run_mini_box_equilibration(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    timestep_ps: float,
    temp: float,
    echeck: float,
    duration_ps: float,
    use_pbc: bool,
    box_side: float | None,
) -> None:
    """Run a short CPT NPT leg between CHARMM MM mini and MLpot registration."""
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _run_charmm_mm_pretreat_cpt_stage,
    )

    if float(duration_ps) <= 0.0:
        return
    if not use_pbc:
        raise ValueError("mini box equilibration requires PBC")
    if box_side is not None:
        configure_liquid_box_mini_equil_args(args, box_side_A=float(box_side))
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_settings,
    )

    pretreat = resolve_charmm_mm_pretreat_settings(args)
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _pretreat_use_fixed_box_nvt,
    )

    fixed_box = _pretreat_use_fixed_box_nvt(args, use_pbc=use_pbc)
    if not args.quiet:
        mode = (
            f"Hoover NVT at L={float(getattr(args, 'box_size', box_side)):.3f} Å"
            if fixed_box
            else "CPT NPT"
        )
        print(
            f"\nMini box equilibration: {mode} for {float(duration_ps):.2f} ps "
            f"(before MLpot SD)",
            flush=True,
        )
    _run_charmm_mm_pretreat_cpt_stage(
        "equi",
        args,
        paths={
            **paths,
            "charmm_mm_equi_res": paths["mini_box_equil_res"],
            "charmm_mm_equi_dcd": paths["mini_box_equil_dcd"],
        },
        res_key="charmm_mm_equi_res",
        dcd_key="charmm_mm_equi_dcd",
        timestep_ps=pretreat.timestep_ps,
        duration_ps=float(duration_ps),
        temp=pretreat.temperature_K,
        pressure_atm=pretreat.pressure_atm,
        echeck=echeck,
        use_pbc=True,
        box_side=box_side,
        include_firstt=True,
    )
    sync_charmm_positions(get_charmm_positions_array())
