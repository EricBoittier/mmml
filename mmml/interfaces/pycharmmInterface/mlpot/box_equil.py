"""Short CPT NPT box equilibration during the staged mini step."""

from __future__ import annotations

from pathlib import Path

import argparse

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array, sync_charmm_positions

# Lattice ABNR and Hoover mini-equil share the same MM-only stress ceiling.
MAX_MM_PRETREAT_DYNAMICS_GRMS = 500.0
MAX_MM_PRETREAT_COORD_SPAN_A = 500.0


def measure_mm_pretreat_grms() -> float:
    """Fresh CHARMM GRMS (kcal/mol/Å) after ``ENER FORCE``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force

    return float(charmm_grms_after_ener_force(silent=True))


def mm_geometry_safe_for_pretreat_dynamics(
    *,
    grms_kcalmol_A: float,
    positions: np.ndarray | None = None,
    max_grms: float = MAX_MM_PRETREAT_DYNAMICS_GRMS,
    max_coord_span_A: float = MAX_MM_PRETREAT_COORD_SPAN_A,
) -> tuple[bool, str]:
    """Return whether CHARMM MM pretreat dynamics (lattice / mini equil) is safe."""
    if not np.isfinite(float(grms_kcalmol_A)):
        return False, "non-finite CHARMM GRMS after MM minimize"
    grms = float(grms_kcalmol_A)
    if grms > float(max_grms):
        return (
            False,
            f"CHARMM GRMS {grms:.1f} kcal/mol/Å exceeds safe ceiling "
            f"{float(max_grms):.1f} kcal/mol/Å for MM pretreat dynamics",
        )
    if positions is not None:
        pos = np.asarray(positions, dtype=float)
        if pos.size and not np.all(np.isfinite(pos)):
            return False, "non-finite coordinates after MM minimize"
        if pos.size:
            max_abs = float(np.max(np.abs(pos)))
            if max_abs > float(max_coord_span_A):
                return (
                    False,
                    f"coordinate span {max_abs:.1f} Å exceeds "
                    f"{float(max_coord_span_A):.1f} Å before MM pretreat dynamics",
                )
    return True, "ok"


def maybe_run_mini_box_equilibration(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    timestep_ps: float,
    temp: float,
    echeck: float,
    duration_ps: float,
    use_pbc: bool,
    box_side: float | None,
    grms_kcalmol_A: float | None = None,
    positions: np.ndarray | None = None,
) -> bool:
    """Run mini box equil when geometry is safe; otherwise skip with a warning."""
    if float(duration_ps) <= 0.0:
        return False
    grms = (
        float(grms_kcalmol_A)
        if grms_kcalmol_A is not None
        else None
    )
    pos = positions
    if grms is None or pos is None:
        if grms is None:
            grms = measure_mm_pretreat_grms()
        if pos is None:
            pos = get_charmm_positions_array()
    safe, reason = mm_geometry_safe_for_pretreat_dynamics(
        grms_kcalmol_A=grms,
        positions=pos,
    )
    if not safe:
        if not getattr(args, "quiet", False):
            print(
                f"Mini box equilibration: skipped — {reason}. "
                "Rebuild with a looser density (--profile conservative, lower "
                "--bulk-density-fraction, or larger --box-size).",
                flush=True,
            )
        return False
    run_mini_box_equilibration(
        args,
        paths=paths,
        timestep_ps=timestep_ps,
        temp=temp,
        echeck=echeck,
        duration_ps=float(duration_ps),
        use_pbc=use_pbc,
        box_side=box_side,
    )
    return True


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
    grms = measure_mm_pretreat_grms()
    pos = get_charmm_positions_array()
    safe, reason = mm_geometry_safe_for_pretreat_dynamics(
        grms_kcalmol_A=grms,
        positions=pos,
    )
    if not safe:
        if not getattr(args, "quiet", False):
            print(
                f"Mini box equilibration: skipped — {reason}. "
                "Rebuild with a looser density (--profile conservative, lower "
                "--bulk-density-fraction, or larger --box-size).",
                flush=True,
            )
        return
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
