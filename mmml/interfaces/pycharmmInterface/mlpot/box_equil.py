"""Short CPT NPT box equilibration during the staged mini step."""

from __future__ import annotations

from pathlib import Path

import argparse

from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array, sync_charmm_positions


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
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_settings,
    )

    pretreat = resolve_charmm_mm_pretreat_settings(args)
    if not args.quiet:
        print(
            f"\nMini box equilibration: CPT NPT for {float(duration_ps):.2f} ps "
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
