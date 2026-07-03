"""MM pretreat box equilibration (hot→cold cycle) during the staged mini step."""

from __future__ import annotations

from pathlib import Path

import argparse

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array, sync_charmm_positions

# Lattice ABNR and Hoover mini-equil share the same MM-only stress ceiling.
MAX_MM_PRETREAT_DYNAMICS_GRMS = 500.0
MAX_MM_PRETREAT_COORD_SPAN_A = 500.0
DEFAULT_MINI_BOX_EQUIL_PS = 200.0


def measure_mm_pretreat_grms() -> float:
    """Fresh CHARMM GRMS (kcal/mol/Å) after ``ENER FORCE``."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force

    return float(charmm_grms_after_ener_force(silent=True))


def resolve_mini_box_equil_hot_temp_K(args: argparse.Namespace, *, target_K: float) -> float:
    """Peak temperature for the pretreat hot leg (K)."""
    raw = getattr(args, "mini_box_equil_hot_temp", None)
    if raw is not None:
        hot = float(raw)
    else:
        hot = max(float(target_K) * 1.5, float(target_K) + 100.0)
    if hot <= float(target_K):
        raise ValueError(
            f"mini_box_equil_hot_temp must exceed target temperature ({target_K:.1f} K), got {hot:.1f} K"
        )
    return hot


def resolve_mini_box_equil_durations_ps(
    args: argparse.Namespace,
    *,
    duration_ps: float | None = None,
) -> tuple[float, float] | None:
    """Return (ps_heat, ps_cool) for the pretreat hot→cold cycle, or None when off."""
    total = float(
        duration_ps
        if duration_ps is not None
        else getattr(args, "mini_box_equil_ps", 0.0) or 0.0
    )
    if total <= 0.0:
        return None
    ps_heat_raw = getattr(args, "mini_box_equil_ps_heat", None)
    ps_cool_raw = getattr(args, "mini_box_equil_ps_cool", None)
    if ps_heat_raw is not None and ps_cool_raw is not None:
        ps_heat = float(ps_heat_raw)
        ps_cool = float(ps_cool_raw)
    elif ps_heat_raw is not None:
        ps_heat = float(ps_heat_raw)
        ps_cool = max(0.0, total - ps_heat)
    elif ps_cool_raw is not None:
        ps_cool = float(ps_cool_raw)
        ps_heat = max(0.0, total - ps_cool)
    else:
        ps_heat = total / 2.0
        ps_cool = total / 2.0
    if ps_heat <= 0.0 and ps_cool <= 0.0:
        return None
    return ps_heat, ps_cool


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
    if resolve_mini_box_equil_durations_ps(args, duration_ps=duration_ps) is None:
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


def _run_mini_box_equil_heat_leg(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    res_key: str,
    dcd_key: str,
    timestep_ps: float,
    duration_ps: float,
    firstt: float,
    finalt: float,
    echeck: float,
    use_pbc: bool,
    coords_in_memory: bool,
    restart_read_key: str | None = None,
    overlap_context: str,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_dynamics_print_kwargs,
        apply_pretreat_dyn_freq_kwargs,
        resolve_dcd_nsavc,
        resolve_heat_ihtfrq,
        resolve_pretreat_dynamics_print_kwargs,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        apply_heat_ramp_frequencies,
        build_heat_dynamics,
        run_dynamics_with_io,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _configure_heat_dynamics_start,
        _reset_stage_trajectory,
    )

    nstep = max(1, int(round(float(duration_ps) / float(timestep_ps))))
    heat_echeck = echeck
    if getattr(args, "no_echeck", False) or getattr(args, "no_echeck_heat", False):
        heat_echeck = -1.0
    save = bool(getattr(args, "save", True))
    dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=getattr(args, "dcd_nsavc", None), nstep=nstep)
    save_interval_ps = timestep_ps * max(1, dcd_nsavc)
    kw = build_heat_dynamics(
        timestep_ps=timestep_ps,
        duration_ps=float(duration_ps),
        save_interval_ps=save_interval_ps,
        temp=float(finalt),
        firstt=float(firstt),
        finalt=float(finalt),
        echeck=heat_echeck,
        use_pbc=use_pbc,
        ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep),
    )
    dyn_print = resolve_pretreat_dynamics_print_kwargs(nstep=nstep)
    kw["nstep"] = nstep
    apply_dynamics_print_kwargs(kw, dyn_print)
    kw["iasors"] = 0
    apply_heat_ramp_frequencies(
        kw, nstep=nstep, ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep)
    )
    apply_pretreat_dyn_freq_kwargs(
        kw,
        args,
        use_pbc=use_pbc,
        dt_fs=float(timestep_ps) * 1000.0,
    )
    restart_from_file = (
        not coords_in_memory
        and restart_read_key is not None
        and Path(paths[restart_read_key]).is_file()
    )
    io = CharmmTrajectoryFiles(
        restart_read=paths.get(restart_read_key) if restart_from_file else None,
        restart_write=paths[res_key],
        trajectory=paths.get(dcd_key) if save else None,
    )
    _configure_heat_dynamics_start(
        kw,
        io,
        coords_in_memory=coords_in_memory,
        restart_from_file=restart_from_file,
        timestep_ps=timestep_ps,
        use_pbc=use_pbc,
        quiet=True,
        heat_thermostat="scale",
    )
    if save and io.trajectory is not None:
        _reset_stage_trajectory(
            Path(io.trajectory),
            rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
        )
    run_dynamics_with_io(
        kw,
        io,
        overlap=None,
        overlap_context=overlap_context,
        mlpot_ctx=None,
        rng_base=getattr(args, "seed", None),
    )
    if save:
        assert_stage_dynamics_completed(
            stage=overlap_context.lower(),
            expected_nstep=nstep,
            nsavc=dcd_nsavc,
            dcd_path=paths.get(dcd_key),
            restart_path=paths.get(res_key),
            allow_incomplete=bool(getattr(args, "allow_incomplete_dynamics", False)),
        )


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
    """Run MM hot→cold pretreat dynamics between CHARMM MM mini and MLpot registration."""
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_quiet_output
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        resolve_charmm_mm_pretreat_settings,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
        _pretreat_use_fixed_box_nvt,
    )

    durations = resolve_mini_box_equil_durations_ps(args, duration_ps=duration_ps)
    if durations is None:
        return
    ps_heat, ps_cool = durations
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

    pretreat = resolve_charmm_mm_pretreat_settings(args)
    target_K = float(temp if temp is not None else pretreat.temperature_K)
    hot_K = resolve_mini_box_equil_hot_temp_K(args, target_K=target_K)
    fixed_box = _pretreat_use_fixed_box_nvt(args, use_pbc=use_pbc)
    total_ps = ps_heat + ps_cool
    if not args.quiet:
        mode = (
            f"Hoover NVT at L={float(getattr(args, 'box_size', box_side)):.3f} Å"
            if fixed_box
            else "CPT NPT"
        )
        print(
            f"\nMini box equilibration: {mode} hot→cold "
            f"{target_K:.0f}→{hot_K:.0f}→{target_K:.0f} K, "
            f"{ps_heat:.1f}+{ps_cool:.1f}={total_ps:.1f} ps (before MLpot SD)",
            flush=True,
        )

    pretreat_dir = Path(paths["mini_box_equil_res"]).parent
    pretreat_dir.mkdir(parents=True, exist_ok=True)
    leg_paths = {
        **paths,
        "mini_box_equil_hot_res": pretreat_dir / "mini_box_equil_hot.res",
        "mini_box_equil_hot_dcd": pretreat_dir / "mini_box_equil_hot.dcd",
    }

    with charmm_quiet_output():
        apply_charmm_mm_block()
        if ps_heat > 0.0:
            _run_mini_box_equil_heat_leg(
                args,
                paths=leg_paths,
                res_key="mini_box_equil_hot_res",
                dcd_key="mini_box_equil_hot_dcd",
                timestep_ps=pretreat.timestep_ps,
                duration_ps=float(ps_heat),
                firstt=target_K,
                finalt=hot_K,
                echeck=echeck,
                use_pbc=use_pbc,
                coords_in_memory=True,
                restart_read_key=None,
                overlap_context="MINI_BOX_EQUIL_HOT",
            )
            sync_charmm_positions(get_charmm_positions_array())
        if ps_cool > 0.0:
            _run_mini_box_equil_heat_leg(
                args,
                paths=paths,
                res_key="mini_box_equil_res",
                dcd_key="mini_box_equil_dcd",
                timestep_ps=pretreat.timestep_ps,
                duration_ps=float(ps_cool),
                firstt=hot_K if ps_heat > 0.0 else target_K,
                finalt=target_K,
                echeck=echeck,
                use_pbc=use_pbc,
                coords_in_memory=True,
                restart_read_key="mini_box_equil_hot_res" if ps_heat <= 0.0 else None,
                overlap_context="MINI_BOX_EQUIL_COLD",
            )
            sync_charmm_positions(get_charmm_positions_array())
