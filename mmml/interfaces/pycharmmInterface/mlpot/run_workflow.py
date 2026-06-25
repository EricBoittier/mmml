"""In-process CHARMM MLpot minimize / dynamics workflows (``mmml md-system --backend pycharmm``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np

PathLike = str | Path

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_charmm_output_from_args,
    build_cluster_from_args_with_tag,
    dynamics_nstep_from_ps,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_constrain_resids,
    resolve_dcd_nsavc,
    resolve_dynamics_print_kwargs,
    resolve_heat_ihtfrq,
    resolve_heat_firstt_finalt,
    apply_flat_bottom_from_args,
    assert_dynamics_ready,
    charmm_grms,
    resolve_echeck_for_cluster,
    resolve_mini_nstep,
    refresh_mlpot_energy_and_grms,
    resolve_fix_resids,
    resolve_pbc_box_side,
    resolve_show_energy,
    resolve_test_first_config,
    resolve_charmm_use_pbc,
    resolve_loose_pbc,
    resolve_mlpot_use_pbc,
    resolve_use_pbc,
    setup_cons_fix_for_resids,
    timestep_ps_from_dt_fs,
    turn_off_cons_fix,
    validate_resids_for_cluster,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmMmMinimizeConfig,
    CharmmTrajectoryFiles,
    MinimizeWithMlpotConfig,
    apply_heat_ramp_frequencies,
    assign_velocities_at_temperature,
    build_heat_dynamics,
    build_nve_dynamics,
    minimize_charmm_mm_only,
    minimize_with_mlpot,
    run_dynamics_with_io,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    apply_comp_velocity_policy,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    assert_mlpot_user_active,
    verify_mlpot_charmm_atom_consistency,
    get_charmm_positions_array,
    load_physnet_mlpot_bundle,
    refresh_nbonds_after_mlpot_pbc,
    register_mlpot,
    save_cluster_topology_for_vmd,
    select_all_atoms,
    select_by_resids,
    setup_default_nbonds,
    sync_charmm_positions,
)
from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

Phase = Literal["minimize", "dynamics", "full", "staged"]
Ensemble = Literal["nve", "nvt"]
dcd_nsavc = 100

def _charmm_pre_minimize_before_mlpot(
    args: argparse.Namespace,
    *,
    nprint: int,
    reference_positions: np.ndarray | None = None,
    dcd_path: Path | None = None,
    dcd_nsavc: int = 1,
    save_crd_path: Path | None = None,
    save_pdb_path: Path | None = None,
    save_psf_path: Path | None = None,
    save_energy_json_path: Path | None = None,
    save_title: str = "CHARMM MM pre-minimize",
    use_pbc: bool = False,
) -> np.ndarray:
    """CGENFF SD/ABNR on the built cluster before :func:`register_mlpot`."""
    if not getattr(args, "charmm_pre_minimize", True):
        return get_charmm_positions_array()

    n_sd = int(getattr(args, "charmm_sd_steps", 50))
    n_abnr = int(getattr(args, "charmm_abnr_steps", 100))
    tolenr = float(getattr(args, "charmm_tolenr", 1e-3))
    tolgrd = float(getattr(args, "charmm_tolgrd", 1e-3))
    if not args.quiet:
        print(f"\nCHARMM MM pre-minimize: SD={n_sd} ABNR={n_abnr}", flush=True)
    minimize_charmm_mm_only(
        CharmmMmMinimizeConfig(
            nstep_sd=n_sd,
            nstep_abnr=n_abnr,
            nprint=nprint,
            tolenr=tolenr,
            tolgrd=tolgrd,
            verbose=not args.quiet,
            show_energy=resolve_show_energy(args),
            reference_positions=reference_positions,
            dcd_path=dcd_path,
            dcd_nsavc=dcd_nsavc,
            save_crd_path=save_crd_path,
            save_pdb_path=save_pdb_path,
            save_psf_path=save_psf_path,
            save_energy_json_path=save_energy_json_path,
            save_title=save_title,
            use_pbc=use_pbc,
        )
    )
    r_mm = get_charmm_positions_array()
    grms = charmm_grms()
    if not args.quiet:
        print(f"Post MM pre-min GRMS: {grms:.4f} kcal/mol/Å", flush=True)
    return r_mm


def _resolve_charmm_mm_pretreat_heat_nstep(args: argparse.Namespace, *, timestep_ps: float) -> int:
    ps_heat = getattr(args, "charmm_mm_pretreat_ps_heat", None)
    if ps_heat is not None and float(ps_heat) > 0.0:
        return max(1, dynamics_nstep_from_ps(float(ps_heat), float(getattr(args, "dt_fs", 0.25))))
    return max(1, int(getattr(args, "charmm_mm_pretreat_heat_nstep", 2000)))


def _pretreat_use_fixed_box_nvt(args: argparse.Namespace, *, use_pbc: bool) -> bool:
    """Pretreat equi/prod at explicit ``--box-size`` use Hoover NVT, not CPT NPT."""
    if not use_pbc:
        return False
    raw = getattr(args, "box_size", None)
    if raw is None:
        return False
    try:
        return float(raw) > 0.0
    except (TypeError, ValueError):
        return False


def _run_charmm_mm_pretreat_cpt_stage(
    stage: Literal["equi", "prod"],
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    res_key: str,
    dcd_key: str,
    timestep_ps: float,
    duration_ps: float,
    temp: float,
    echeck: float,
    use_pbc: bool,
    box_side: float | None,
    include_firstt: bool,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        build_cpt_equilibration_dynamics,
        build_cpt_production_dynamics,
        build_nvt_equilibration_dynamics,
        build_nvt_production_dynamics,
        ps_to_nsteps,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _configure_npt_dynamics_start,
        _reset_stage_trajectory,
        _seed_restart_for_memory_handoff,
        _npt_cpt_options,
    )

    nstep = max(1, ps_to_nsteps(timestep_ps, duration_ps))
    save = bool(getattr(args, "save", True))
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
        timestep_ps=timestep_ps,
        nstep=nstep,
    )
    save_interval_ps = timestep_ps * max(1, dcd_nsavc)
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep, nsavc=dcd_nsavc)
    stage_echeck = max(echeck, 500.0) if echeck > 0 else echeck
    use_cpt = use_pbc and not _pretreat_use_fixed_box_nvt(args, use_pbc=use_pbc)

    if stage == "equi":
        if use_cpt:
            kw = build_cpt_equilibration_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=False,
                echeck=stage_echeck,
                include_firstt=include_firstt,
                **_npt_cpt_options(args),
            )
        else:
            kw = build_nvt_equilibration_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=False,
                echeck=stage_echeck,
                include_firstt=include_firstt,
            )
    else:
        if use_cpt:
            kw = build_cpt_production_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=False,
                echeck=stage_echeck,
                **_npt_cpt_options(args),
            )
        else:
            kw = build_nvt_production_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=False,
                echeck=stage_echeck,
            )

    kw["nstep"] = nstep
    kw["nprint"] = dyn_print["nprint"]
    kw["iprfrq"] = dyn_print["iprfrq"]
    kw["isvfrq"] = dyn_print["isvfrq"]
    kw["new"] = False
    kw["start"] = False
    kw["restart"] = False

    io = CharmmTrajectoryFiles(
        restart_write=paths[res_key],
        trajectory=paths[dcd_key] if save else None,
    )
    if save and io.trajectory is not None:
        _reset_stage_trajectory(
            Path(io.trajectory),
            rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
        )
    if not args.quiet:
        print(
            f"CHARMM MM pretreat {stage}: {duration_ps:.2f} ps "
            f"({nstep} steps @ {timestep_ps} ps) | memory handoff",
            flush=True,
        )
    _seed_restart_for_memory_handoff(io, kw, stage=stage)
    _configure_npt_dynamics_start(
        kw,
        io,
        coords_in_memory=True,
        restart_from_file=False,
        timestep_ps=timestep_ps,
        use_pbc=use_pbc,
        quiet=bool(args.quiet),
        temp=temp,
        box_side=box_side,
    )
    run_dynamics_with_io(
        kw,
        io,
        overlap=None,
        overlap_context=f"CHARMM_MM_PRETREAT_{stage.upper()}",
        mlpot_ctx=None,
        rng_base=getattr(args, "seed", None),
    )
    if save:
        assert_stage_dynamics_completed(
            stage=f"charmm_mm_{stage}",
            expected_nstep=nstep,
            nsavc=dcd_nsavc,
            dcd_path=paths.get(dcd_key),
            restart_path=paths.get(res_key),
            allow_incomplete=bool(getattr(args, "allow_incomplete_dynamics", False)),
        )
    sync_charmm_positions(get_charmm_positions_array())
    if not args.quiet:
        print(
            f"CHARMM MM pretreat {stage} done -> {paths[res_key]}",
            flush=True,
        )


def run_charmm_mm_pretreat_before_mlpot(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    timestep_ps: float,
    use_pbc: bool,
    temp: float,
    echeck: float,
    mini_nprint: int,
    reference_positions: np.ndarray | None = None,
    skip_minimize: bool = False,
) -> np.ndarray:
    """CGENFF minimize + CHARMM heat/equi/prod on the built cluster before :func:`register_mlpot`.

    Uses full MM terms (``apply_charmm_mm_block``), not MLpot USER. Intended for
    relaxing Packmol clashes and classical MD before ML dynamics. Set
    ``charmm_mm_pretreat_ps_equi`` / ``charmm_mm_pretreat_ps_prod`` > 0 for equilibration
    and production (CPT NPT when ``use_pbc`` and no ``--box-size``; fixed ``--box-size``
    uses Hoover NVT at that L).

    When ``skip_minimize`` is true (handoff continuations), skip CGENFF SD/ABNR and
    run heat/equi/prod on coordinates already in CHARMM memory.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        resume_charmm_mm_pretreat_if_available,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_charmm_state_from_restart,
    )

    n_heat = _resolve_charmm_mm_pretreat_heat_nstep(args, timestep_ps=timestep_ps)
    ps_equi = float(getattr(args, "charmm_mm_pretreat_ps_equi", 0.0) or 0.0)
    ps_prod = float(getattr(args, "charmm_mm_pretreat_ps_prod", 0.0) or 0.0)
    pretreat_resume = resume_charmm_mm_pretreat_if_available(
        paths,
        args,
        timestep_ps=timestep_ps,
    )
    if pretreat_resume.skip_entire_pretreat and pretreat_resume.restart_read is not None:
        restore_charmm_state_from_restart(pretreat_resume.restart_read)
        if not args.quiet:
            print(
                "CHARMM MM pretreat: skip completed legs "
                f"(prod restart {pretreat_resume.restart_read.name})",
                flush=True,
            )
        return get_charmm_positions_array()
    n_sd = int(
        getattr(args, "charmm_mm_pretreat_mini_sd", None)
        if getattr(args, "charmm_mm_pretreat_mini_sd", None) is not None
        else getattr(args, "charmm_sd_steps", 50)
    )
    n_abnr = int(
        getattr(args, "charmm_mm_pretreat_mini_abnr", None)
        if getattr(args, "charmm_mm_pretreat_mini_abnr", None) is not None
        else getattr(args, "charmm_abnr_steps", 100)
    )
    if not args.quiet:
        mini_label = "skip mini" if skip_minimize else f"SD={n_sd} ABNR={n_abnr}"
        print(
            f"\nCHARMM MM pretreat (no MLpot): {mini_label}, heat nstep={n_heat}"
            + (f", equi={ps_equi:.2f} ps" if ps_equi > 0.0 else "")
            + (f", prod={ps_prod:.2f} ps" if ps_prod > 0.0 else ""),
            flush=True,
        )

    apply_charmm_mm_block()
    if not use_pbc:
        setup_default_nbonds()

    if pretreat_resume.restart_read is not None and not (
        pretreat_resume.heat_integrated_step > 0 and not pretreat_resume.skip_heat
    ):
        restore_charmm_state_from_restart(pretreat_resume.restart_read)
        if not args.quiet:
            print(
                f"CHARMM MM pretreat: resumed from {pretreat_resume.restart_read.name}",
                flush=True,
            )

    save = bool(getattr(args, "save", True))
    pretreat_dir = paths["charmm_mm_heat_res"].parent
    pretreat_dir.mkdir(parents=True, exist_ok=True)
    heat_dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=n_heat)
    skip_minimize = bool(skip_minimize or pretreat_resume.skip_minimize)
    if not skip_minimize:
        minimize_charmm_mm_only(
            CharmmMmMinimizeConfig(
                nstep_sd=n_sd,
                nstep_abnr=n_abnr,
                nprint=mini_nprint,
                tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                verbose=not args.quiet,
                show_energy=resolve_show_energy(args),
                reference_positions=reference_positions,
                dcd_path=paths.get("mini_charmm_dcd") if save else None,
                dcd_nsavc=resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=max(n_sd, 1))
                if save
                else 0,
                use_pbc=use_pbc,
            )
        )
        if not args.quiet:
            print(
                f"CHARMM MM pretreat post-min GRMS: {charmm_grms():.4f} kcal/mol/Å",
                flush=True,
            )
    elif not args.quiet:
        print(
            "CHARMM MM pretreat: skip CGENFF mini (handoff coords in memory)",
            flush=True,
        )

    if not pretreat_resume.skip_heat:
        heat_firstt, heat_finalt = resolve_heat_firstt_finalt(args, default_temp=temp)
        heat_integrated = max(0, int(pretreat_resume.heat_integrated_step))
        n_heat_run = max(1, n_heat - heat_integrated) if heat_integrated > 0 else n_heat
        save_interval_ps = timestep_ps * max(
            1,
            resolve_dcd_nsavc(
                dcd_nsavc=args.dcd_nsavc,
                dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
                timestep_ps=timestep_ps,
                nstep=n_heat_run,
            ),
        )
        kw = build_heat_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=n_heat_run * timestep_ps,
            save_interval_ps=save_interval_ps,
            temp=temp,
            firstt=heat_firstt,
            finalt=heat_finalt,
            echeck=echeck,
            use_pbc=use_pbc,
            ihtfrq=resolve_heat_ihtfrq(args, nstep=n_heat_run),
        )
        dyn_print = resolve_dynamics_print_kwargs(args, nstep=n_heat_run, nsavc=dcd_nsavc)
        kw["nstep"] = n_heat_run
        kw["nprint"] = dyn_print["nprint"]
        kw["iprfrq"] = dyn_print["iprfrq"]
        kw["isvfrq"] = dyn_print["isvfrq"]
        kw["iasors"] = 0
        kw["iasvel"] = 1
        apply_heat_ramp_frequencies(
            kw, nstep=n_heat_run, ihtfrq=resolve_heat_ihtfrq(args, nstep=n_heat_run)
        )

        io = CharmmTrajectoryFiles(
            restart_read=paths["charmm_mm_heat_res"] if heat_integrated > 0 else None,
            restart_write=paths["charmm_mm_heat_res"],
            trajectory=paths["charmm_mm_heat_dcd"] if save else None,
        )
        from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
            _configure_heat_dynamics_start,
            _reset_stage_trajectory,
        )

        if heat_integrated == 0:
            assign_velocities_at_temperature(
                float(heat_firstt),
                timestep_ps=timestep_ps,
                restart_path=None,
                use_pbc=use_pbc,
            )
            kw["restart"] = False
            kw["new"] = False
            kw["start"] = False
        else:
            _configure_heat_dynamics_start(
                kw,
                io,
                coords_in_memory=False,
                restart_from_file=True,
                timestep_ps=timestep_ps,
                use_pbc=use_pbc,
                quiet=bool(args.quiet),
                heat_thermostat="scale",
            )

        if save and io.trajectory is not None and heat_integrated == 0:
            _reset_stage_trajectory(
                Path(io.trajectory),
                rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
            )
        if not args.quiet:
            if heat_integrated > 0:
                print(
                    f"CHARMM MM pretreat heat: resume step {heat_integrated}/{n_heat}, "
                    f"{n_heat_run} steps remaining | "
                    f"{heat_firstt:.1f} -> {heat_finalt:.1f} K @ {timestep_ps} ps",
                    flush=True,
                )
            else:
                print(
                    f"CHARMM MM pretreat heat: {heat_firstt:.1f} -> {heat_finalt:.1f} K, "
                    f"{n_heat_run} steps @ {timestep_ps} ps | ihtfrq={kw.get('ihtfrq')}",
                    flush=True,
                )
        run_dynamics_with_io(
            kw,
            io,
            overlap=None,
            overlap_context="CHARMM_MM_PRETREAT_HEAT",
            mlpot_ctx=None,
            rng_base=getattr(args, "seed", None),
        )
        if save:
            assert_stage_dynamics_completed(
                stage="charmm_mm_heat",
                expected_nstep=n_heat,
                nsavc=heat_dcd_nsavc,
                dcd_path=paths.get("charmm_mm_heat_dcd"),
                restart_path=paths.get("charmm_mm_heat_res"),
                allow_incomplete=bool(getattr(args, "allow_incomplete_dynamics", False)),
            )
        sync_charmm_positions(get_charmm_positions_array())
        if not args.quiet:
            print(
                f"CHARMM MM pretreat heat done -> {paths['charmm_mm_heat_res']}",
                flush=True,
            )
    elif not args.quiet:
        print("CHARMM MM pretreat: skip heat (restart on disk)", flush=True)

    box_side = resolve_pbc_box_side(args, get_charmm_positions_array()) if use_pbc else None
    use_fixed_nvt = _pretreat_use_fixed_box_nvt(args, use_pbc=use_pbc)
    if use_pbc and not args.quiet and (ps_equi > 0.0 or ps_prod > 0.0):
        if use_fixed_nvt:
            print(
                f"CHARMM MM pretreat equi/prod: fixed box L={box_side:.3f} Å "
                "(Hoover NVT; box size matches campaign --box-size)",
                flush=True,
            )
        else:
            print(
                "CHARMM MM pretreat equi/prod: CPT NPT (box may drift; prod preserves equi box)",
                flush=True,
            )
    if ps_equi > 0.0 and not pretreat_resume.skip_equi:
        _run_charmm_mm_pretreat_cpt_stage(
            "equi",
            args,
            paths=paths,
            res_key="charmm_mm_equi_res",
            dcd_key="charmm_mm_equi_dcd",
            timestep_ps=timestep_ps,
            duration_ps=ps_equi,
            temp=temp,
            echeck=echeck,
            use_pbc=use_pbc,
            box_side=box_side,
            include_firstt=False,
        )
    elif ps_equi > 0.0 and not args.quiet:
        print("CHARMM MM pretreat: skip equi (restart on disk)", flush=True)
    if ps_prod > 0.0:
        # After NPT equi the live box differs from --box-size; do not reset crystal to config L.
        prod_box_side = box_side if (use_fixed_nvt or ps_equi <= 0.0) else None
        _run_charmm_mm_pretreat_cpt_stage(
            "prod",
            args,
            paths=paths,
            res_key="charmm_mm_prod_res",
            dcd_key="charmm_mm_prod_dcd",
            timestep_ps=timestep_ps,
            duration_ps=ps_prod,
            temp=temp,
            echeck=echeck,
            use_pbc=use_pbc,
            box_side=prod_box_side,
            include_firstt=False,
        )

    if not args.quiet and (ps_equi > 0.0 or ps_prod > 0.0):
        print("CHARMM MM pretreat complete (CGENFF mini + heat/equi/prod)", flush=True)
    elif not args.quiet:
        print(
            f"CHARMM MM pretreat done -> {paths['charmm_mm_heat_res']}",
            flush=True,
        )
    return get_charmm_positions_array()


def _atoms_per_monomer_list(z: np.ndarray, n_monomers: int) -> list[int]:
    n_atoms = int(len(z))
    if n_atoms % int(n_monomers) != 0:
        raise ValueError(
            f"atom count {n_atoms} not divisible by n_monomers={n_monomers}"
        )
    per = n_atoms // int(n_monomers)
    return [per] * int(n_monomers)


def _setup_charmm_nbonds_for_args(
    args: argparse.Namespace,
    r: np.ndarray,
) -> float | None:
    """Vacuum or PBC CHARMM environment; return cubic box side when PBC is active."""
    charmm_pbc = resolve_charmm_use_pbc(args)
    mlpot_pbc = resolve_mlpot_use_pbc(args)
    if not charmm_pbc:
        setup_default_nbonds()
        return None
    box_side = resolve_pbc_box_side(args, r)
    if not args.quiet:
        if charmm_pbc and not mlpot_pbc:
            print(
                f"CHARMM loose PBC: cubic L={box_side:.3f} Å "
                "(ML open boundary; no MIC)",
                flush=True,
            )
        else:
            print(f"PBC cubic box: {box_side:.3f} Å", flush=True)
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=box_side)
    return float(box_side)


def _register_mlpot_context(
    z: np.ndarray,
    r: np.ndarray,
    ckpt: Path,
    n_atoms: int,
    n_monomers: int,
    *,
    atoms_per_monomer: list[int] | None = None,
    ml_batch_size: int | None = None,
    ml_gpu_count: int | None = None,
    ml_max_active_dimers: int | None = None,
    cubic_box_side_A: float | None = None,
    mlpot_use_pbc: bool = False,
    verbose: bool = False,
    args: Any | None = None,
    defer_jax_warmup: bool | None = None,
    topology_psf: PathLike | None = None,
):
    import ase

    ml_cell = float(cubic_box_side_A) if mlpot_use_pbc and cubic_box_side_A is not None else None
    if ml_cell is not None and verbose:
        print(
            f"MLpot MIC PBC: cubic L={ml_cell:.3f} Å",
            flush=True,
        )
    elif cubic_box_side_A is not None and verbose and not mlpot_use_pbc:
        print(
            f"MLpot: open boundary (CHARMM box L={float(cubic_box_side_A):.3f} Å; no MIC)",
            flush=True,
        )

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        charmm_lib_links_mpi,
        defer_jax_warmup_until_after_mlpot_sd,
        mpirun_launch_hint,
    )
    from mmml.interfaces.pycharmmInterface.charmm_mpi import _under_mpirun

    if defer_jax_warmup is None:
        defer_jax_warmup = defer_jax_warmup_until_after_mlpot_sd()

    if verbose and charmm_lib_links_mpi() and not _under_mpirun():
        print(
            "mmml: MPI-linked libcharmm.so without mpirun (serial MLpot). "
            "If MLpot registration segfaults in upinb, use:\n  "
            + mpirun_launch_hint("mmml md-system"),
            flush=True,
        )

    atoms = ase.Atoms(numbers=z, positions=r)
    apm = (
        list(atoms_per_monomer)
        if atoms_per_monomer is not None
        else _atoms_per_monomer_list(z, n_monomers)
    )
    import os
    import sys

    if "jax" not in sys.modules:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["MMML_MLPOT_DEVICE"] = "cpu"
    _, _, pyCModel = load_physnet_mlpot_bundle(
        ckpt,
        n_atoms,
        atoms,
        n_monomers=n_monomers,
        atoms_per_monomer=apm,
        ml_batch_size=ml_batch_size,
        ml_gpu_count=ml_gpu_count,
        ml_max_active_dimers=ml_max_active_dimers,
        cell=ml_cell,
        verbose=verbose,
        args=args,
        defer_jax_until_after_sd=bool(defer_jax_warmup),
    )
    mm_internal_scale = (
        float(getattr(args, "mlpot_mm_internal_scale", 0.0) or 0.0)
        if args is not None
        else 0.0
    )
    if mm_internal_scale > 0.0 and verbose:
        print(
            f"MLpot BLOCK: CGENFF BOND/ANGL/DIHE scale={mm_internal_scale:g} on ML atoms "
            "(ELEC/VDW off; USER=PhysNet)",
            flush=True,
        )
    # Register MLpot in CHARMM *before* JAX GPU warmup. Long XLA compile/autodiff
    # before the first ``upinb`` (MLpot exclusion rebuild) can segfault on MPI
    # CHARMM builds (DCM clusters, PBC NpT).
    ctx = register_mlpot(
        pyCModel,
        z,
        select_all_atoms(),
        use_pbc=mlpot_use_pbc,
        mm_internal_scale=mm_internal_scale,
        cubic_box_side_A=ml_cell,
        verbose=bool(getattr(args, "verbose", False)) if args is not None else False,
    )
    from mmml.interfaces.pycharmmInterface.jax_device_policy import apply_mlpot_jax_platform_env

    apply_mlpot_jax_platform_env(quiet=not verbose)
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        pin_cuda_for_spatial_mpi,
        spatial_mpi_enabled,
    )

    if spatial_mpi_enabled(
        getattr(args, "ml_spatial_mpi", None) if args is not None else None
    ):
        if pin_cuda_for_spatial_mpi() and verbose:
            print(
                f"MLpot spatial MPI: pinned CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
                flush=True,
            )
    if int(n_monomers) > 1:
        from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
            DecomposedMlpotModel,
            warmup_decomposed_mlpot,
        )

        if isinstance(pyCModel, DecomposedMlpotModel):
            if defer_jax_warmup:
                if verbose:
                    print(
                        "Decomposed MLpot: deferring JAX GPU warmup until after MLpot SD "
                        "(MPI-linked CHARMM)",
                        flush=True,
                    )
            else:
                warmup_decomposed_mlpot(
                    pyCModel,
                    r,
                    cell=ml_cell,
                    verbose=verbose,
                )
    ctx.ml_Z = np.asarray(z, dtype=int)
    ctx.use_pbc = bool(mlpot_use_pbc)
    ctx.cubic_box_side_A = float(cubic_box_side_A) if mlpot_use_pbc and cubic_box_side_A else None
    ctx.charmm_cubic_box_side_A = (
        float(cubic_box_side_A) if cubic_box_side_A is not None else None
    )
    if mlpot_use_pbc and cubic_box_side_A is not None:
        # PBC crystal/nbonds were installed in setup_charmm_environment;
        # register_mlpot installs ML exclusions (upinb) before BLOCK and skips a
        # second upinb in MLpot.__init__. Avoid a forced rebuild here too.
        refresh_nbonds_after_mlpot_pbc(
            cubic_box_side_A=float(cubic_box_side_A),
            force=False,
        )
    sync_charmm_positions(r)
    pos_chk = get_charmm_positions_array()
    if np.allclose(pos_chk, 0.0):
        sync_charmm_positions(r)
    from mmml.interfaces.pycharmmInterface.mlpot.topology_recovery import (
        attach_topology_recovery_state,
    )

    attach_topology_recovery_state(ctx, topology_psf)
    return ctx, pyCModel


def sync_mlpot_pbc_cell_from_charmm(
    pyCModel: Any,
    *,
    fallback_side_A: float | None = None,
    restart_path: PathLike | None = None,
    verbose: bool = False,
) -> float:
    """Set ML MIC cell side to the current CHARMM cubic box (NpT / CPT stages).

    Updates the JAX MIC cell only. Does **not** re-run ``prepare_charmm_pbc`` or
    ``update_bnbnd`` here — rebuilding crystal/nbonds with MLpot registered can
    segfault in CHARMM (``upinb``). CPT stage restarts restore CHARMM PBC state.

    When ``pbound_get_size`` is zero (common before a CPT restart is read), pass
    ``restart_path`` to the upcoming/previous ``.res`` file — CHARMM stores the
    box under ``!CRYSTAL PARAMETERS``.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotModel
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_mlpot_mic_box_side_A

    if restart_path is not None:
        rpath = Path(restart_path)
        if isinstance(pyCModel, DecomposedMlpotModel):
            pyCModel._npt_restart_read = rpath if rpath.is_file() else None

    if fallback_side_A is None:
        old_cell = getattr(pyCModel, "_cell", False)
        if old_cell:
            fallback_side_A = float(old_cell)

    side, source = resolve_mlpot_mic_box_side_A(
        fallback_side_A=fallback_side_A,
        restart_path=restart_path,
    )
    old = getattr(pyCModel, "_cell", False)
    if isinstance(pyCModel, DecomposedMlpotModel):
        pyCModel._cell = side
    if verbose:
        if source == "restart":
            print(
                f"MLpot MIC PBC: L={side:.3f} Å from restart "
                f"{Path(restart_path).name if restart_path else ''}",
                flush=True,
            )
        elif source == "fallback":
            print(
                f"MLpot MIC PBC: using L={side:.3f} Å from last known cell "
                f"(CHARMM box query unavailable; CPT restart restores PBC)",
                flush=True,
            )
        elif old and abs(float(old) - side) > 1e-4:
            print(
                f"MLpot MIC PBC synced to CHARMM L={side:.3f} Å "
                f"(was {float(old):.3f} Å)",
                flush=True,
            )
        elif not old:
            print(f"MLpot MIC PBC synced to CHARMM L={side:.3f} Å", flush=True)
    return side


def run_minimize_workflow(args: argparse.Namespace) -> int:
    fix_resids = resolve_fix_resids(args)
    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_mol, tag = build_cluster_from_args_with_tag(args)
    validate_resids_for_cluster(fix_resids, n_mol)
    print_cluster_geometry_summary(r, n_mol)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_atoms = len(z)
    nprint = apply_charmm_output_from_args(args)
    mini_nstep = int(getattr(args, "mini_nstep", getattr(args, "nstep", 10)))
    dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=mini_nstep)

    pdb_path = out_dir / f"mini_full_mlpot_{tag}.pdb"
    crd_path = out_dir / f"mini_full_mlpot_{tag}.crd"
    psf_path = out_dir / f"mini_full_mlpot_{tag}.psf"
    energy_json_path = out_dir / f"mini_full_mlpot_{tag}_energy.json"
    xyz_path = out_dir / f"mini_full_mlpot_{tag}.xyz"
    dcd_path = out_dir / f"mini_full_mlpot_{tag}.dcd"
    charmm_dcd_path = out_dir / f"mini_charmm_mm_{tag}.dcd"
    save = bool(getattr(args, "save", True))

    mlpot_pbc = resolve_mlpot_use_pbc(args)
    charmm_pbc = resolve_charmm_use_pbc(args)
    box_side = _setup_charmm_nbonds_for_args(args, r)
    sync_charmm_positions(r)
    vmd_topo_psf = out_dir / f"cluster_for_vmd_{tag}.psf"
    if not getattr(args, "no_save_vmd_topology", False):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]

    r = _charmm_pre_minimize_before_mlpot(
        args,
        nprint=nprint,
        reference_positions=r,
        dcd_path=charmm_dcd_path if save else None,
        dcd_nsavc=dcd_nsavc if save else 0,
        use_pbc=charmm_pbc,
    )
    sync_charmm_positions(r)

    ctx, pyCModel = _register_mlpot_context(
        z,
        r,
        ckpt,
        n_atoms,
        n_mol,
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_gpu_count=getattr(args, "ml_gpu_count", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        cubic_box_side_A=box_side,
        mlpot_use_pbc=mlpot_pbc,
        verbose=not args.quiet,
        args=args,
        topology_psf=vmd_topo_psf if vmd_topo_psf.is_file() else None,
    )
    fix_sel = select_by_resids(fix_resids) if fix_resids else None
    try:
        ran = minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=fix_sel,
                nstep=mini_nstep,
                nprint=nprint,
                verbose=not args.quiet,
                reference_positions=r,
                pyCModel=pyCModel,
                mlpot_ctx=ctx,
                save=save,
                pdb_path=pdb_path if save else None,
                crd_path=crd_path if save else None,
                psf_path=psf_path if save else None,
                energy_json_path=energy_json_path if save else None,
                xyz_path=xyz_path if save else None,
                dcd_path=dcd_path if save else None,
                dcd_nsavc=dcd_nsavc if save else 0,
                skip_if_crd_exists=False,
                test_first=resolve_test_first_config(args),
            )
        )
    finally:
        ctx.unset()

    print(f"Minimization ran={ran}; artifacts in {out_dir}")
    if save:
        print_vmd_load_help(
            out_dir=out_dir,
            tag=tag,
            topology_psf=vmd_topo_psf,
            trajectory=[charmm_dcd_path, dcd_path],
            n_atoms=n_atoms,
            bondless_psf=psf_path,
        )
    return 0


def run_dynamics_workflow(
    args: argparse.Namespace,
    *,
    ensemble: Ensemble = "nve",
    pre_minimize: bool | None = None,
) -> int:
    fix_resids = resolve_fix_resids(args)
    dynamics_constrain = resolve_constrain_resids(args)
    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_mol, tag = build_cluster_from_args_with_tag(args)
    validate_resids_for_cluster(fix_resids, n_mol)
    validate_resids_for_cluster(dynamics_constrain, n_mol)
    print_cluster_geometry_summary(r, n_mol)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_atoms = len(z)

    dt_fs = float(getattr(args, "dt_fs", 0.25))
    ps = float(getattr(args, "ps", 1.0))
    timestep_ps = timestep_ps_from_dt_fs(dt_fs)
    if getattr(args, "timestep_ps", None) is not None:
        timestep_ps = float(args.timestep_ps)
    nstep = int(getattr(args, "nstep", 0)) or dynamics_nstep_from_ps(ps, dt_fs)
    temp = float(getattr(args, "temperature", getattr(args, "temp", 300.0)))

    mini_nprint = apply_charmm_output_from_args(args)
    echeck = resolve_echeck_for_cluster(args, n_atoms=n_atoms, n_monomers=n_mol)
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        dcd_interval_ps=args.dcd_interval_ps,
        timestep_ps=timestep_ps,
        nstep=nstep,
    )
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep, nsavc=dcd_nsavc)

    stem = "nve" if ensemble == "nve" else "nvt"
    res_path = out_dir / f"{stem}_{tag}.res"
    dcd_path = out_dir / f"{stem}_{tag}.dcd"
    save = bool(getattr(args, "save", True))
    charmm_mini_dcd_path = out_dir / f"mini_charmm_mm_{tag}.dcd"
    mlpot_mini_dcd_path = out_dir / f"mini_full_mlpot_{tag}.dcd"

    if pre_minimize is None:
        pre_minimize = not getattr(args, "no_pre_minimize", False)
    charmm_pbc = resolve_charmm_use_pbc(args)
    mlpot_pbc = resolve_mlpot_use_pbc(args)
    mini_nstep = resolve_mini_nstep(args, n_mol, n_atoms=n_atoms, pbc=mlpot_pbc)
    mini_dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=mini_nstep)
    loose_pbc = resolve_loose_pbc(charmm_pbc, mlpot_pbc)
    box_side = _setup_charmm_nbonds_for_args(args, r)
    overlap_cfg = resolve_dynamics_overlap_config(
        args,
        n_monomers=n_mol,
        use_pbc=charmm_pbc,
        fallback_box_side_A=box_side if charmm_pbc else None,
    )
    sync_charmm_positions(r)
    vmd_topo_psf = out_dir / f"cluster_for_vmd_{tag}.psf"
    if not getattr(args, "no_save_vmd_topology", False):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]

    r = _charmm_pre_minimize_before_mlpot(
        args,
        nprint=mini_nprint,
        reference_positions=r,
        dcd_path=charmm_mini_dcd_path if save else None,
        dcd_nsavc=mini_dcd_nsavc if save else 0,
        use_pbc=charmm_pbc,
    )
    sync_charmm_positions(r)

    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        defer_jax_warmup_until_after_mlpot_sd,
    )

    defer_jax_warmup = defer_jax_warmup_until_after_mlpot_sd()
    ctx, pyCModel = _register_mlpot_context(
        z,
        r,
        ckpt,
        n_atoms,
        n_mol,
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_gpu_count=getattr(args, "ml_gpu_count", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        cubic_box_side_A=box_side,
        mlpot_use_pbc=mlpot_pbc,
        verbose=not args.quiet,
        args=args,
        defer_jax_warmup=defer_jax_warmup,
        topology_psf=vmd_topo_psf if vmd_topo_psf.is_file() else None,
    )
    show_energy = resolve_show_energy(args)
    ml_cell = float(box_side) if mlpot_pbc and box_side is not None else None

    try:
        if pre_minimize:
            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            if not args.quiet:
                print(f"\nMLpot SD minimize: {mini_nstep} steps/pass, {n_atoms} atoms")
            minimize_with_mlpot(
                MinimizeWithMlpotConfig(
                    fixed_ml_selection=fix_sel,
                    nstep=mini_nstep,
                    nprint=mini_nprint,
                    verbose=not args.quiet,
                    reference_positions=r,
                    pyCModel=pyCModel,
                    mlpot_ctx=ctx,
                    save=save,
                    dcd_path=mlpot_mini_dcd_path if save else None,
                    dcd_nsavc=mini_dcd_nsavc if save else 0,
                    show_energy=show_energy,
                    skip_if_crd_exists=False,
                    test_first=resolve_test_first_config(args),
                )
            )
            sync_charmm_positions(get_charmm_positions_array())
            if not args.quiet:
                refresh_mlpot_energy_and_grms(ctx, context="Post MLpot mini")

        if defer_jax_warmup and int(n_mol) > 1:
            from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
                DecomposedMlpotModel,
                warmup_decomposed_mlpot,
            )

            if isinstance(pyCModel, DecomposedMlpotModel):
                warmup_decomposed_mlpot(
                    pyCModel,
                    get_charmm_positions_array(),
                    cell=ml_cell,
                    verbose=not args.quiet,
                )

        # MMFP flat-bottom for dynamics only (avoid fighting SD on the initial Packmol cloud).
        apply_flat_bottom_from_args(args)

        assert_mlpot_user_active(ctx, context="dynamics", quiet=bool(args.quiet))
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
            resolve_max_grms_before_dyn,
        )

        max_grms = resolve_max_grms_before_dyn(
            args,
            n_mol,
            n_atoms,
            pbc=charmm_pbc,
        )
        assert_dynamics_ready(
            max_grms=max_grms,
            abort=not getattr(args, "allow_high_grms", False),
            require_mlpot_user=True,
            mlpot_ctx=ctx,
        )
        verify_mlpot_charmm_atom_consistency(
            ctx,
            expected_z=z,
            context="dynamics",
            quiet=bool(args.quiet),
        )

        if dynamics_constrain:
            setup_cons_fix_for_resids(dynamics_constrain)

        label = ensemble.upper()
        print(
            f"\n{label}: {nstep} steps @ {timestep_ps} ps | "
            f"{format_resid_constraint_message(dynamics_constrain, context='cons_fix')}"
        )
        if show_energy:
            from mmml.interfaces.pycharmmInterface.import_pycharmm import (
                safe_energy_show,
            )

            print("CHARMM energy before dynamics:")
            safe_energy_show()

        io = CharmmTrajectoryFiles(
            restart_write=res_path,
            trajectory=dcd_path,
        )
        save_interval_ps = timestep_ps * dcd_nsavc
        if ensemble == "nve":
            kw = build_nve_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=nstep * timestep_ps,
                save_interval_ps=save_interval_ps,
                restart=False,
                temp=temp,
                nprint=dyn_print["nprint"],
                iprfrq=dyn_print["iprfrq"],
                isvfrq=dyn_print["isvfrq"],
                echeck=echeck,
            )
        else:
            kw = build_heat_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=nstep * timestep_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                echeck=echeck,
                use_pbc=charmm_pbc,
            )
            kw["nprint"] = dyn_print["nprint"]
            kw["iprfrq"] = dyn_print["iprfrq"]
            kw["isvfrq"] = dyn_print["isvfrq"]
            apply_heat_ramp_frequencies(
                kw,
                nstep=nstep,
                ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep),
            )
        kw["new"] = True
        kw["start"] = True
        kw["nstep"] = nstep
        kw["nsavc"] = dcd_nsavc
        print(
            f"DCD nsavc={dcd_nsavc} ({dcd_nsavc * timestep_ps:.6f} ps/frame) | "
            f"dyn print every {dyn_print['nprint']} steps | echeck={echeck} kcal/mol"
        )
        if io.trajectory is not None:
            Path(io.trajectory).unlink(missing_ok=True)
        apply_comp_velocity_policy(
            "nve" if ensemble == "nve" else "heat",
            kw,
            args,
        )
        stage_overlap = overlap_cfg
        if ensemble != "nve" and overlap_cfg is not None:
            from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
                overlap_config_for_stage,
            )

            n_heat_segments = max(1, int(getattr(args, "n_heat_segments", 1)))
            stage_overlap = overlap_config_for_stage(
                overlap_cfg,
                stage="heat",
                nstep=nstep,
                n_segments=n_heat_segments,
            )
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                finalize_heat_dynamics_frequencies,
            )

            finalize_heat_dynamics_frequencies(kw)
        run_dynamics_with_io(
            kw,
            io,
            overlap=stage_overlap,
            overlap_context=label,
            mlpot_ctx=ctx,
            rng_base=getattr(args, "seed", None),
            loose_pbc=loose_pbc,
        )
        if show_energy:
            from mmml.interfaces.pycharmmInterface.import_pycharmm import (
                safe_energy_show,
            )

            print("CHARMM energy after dynamics:")
            safe_energy_show()
    finally:
        if dynamics_constrain:
            turn_off_cons_fix()
        ctx.unset()

    missing = [p for p in (res_path, dcd_path) if not p.is_file()]
    if missing:
        print(f"FAIL: expected outputs missing: {missing}")
        return 1

    print(f"\n{label} OK; restart={res_path.name} trajectory={dcd_path.name} ({out_dir})")
    print_vmd_load_help(
        out_dir=out_dir,
        tag=tag,
        topology_psf=vmd_topo_psf,
        trajectory=[charmm_mini_dcd_path, mlpot_mini_dcd_path, dcd_path],
        n_atoms=n_atoms,
    )
    return 0


def build_charmm_mm_pretreat_handoff_sections(
    positions: np.ndarray,
    *,
    n_monomers: int,
    tag: str,
    pretreat_restart: PathLike | None = None,
    workflow_box_side_A: float | None = None,
    use_pbc: bool = False,
    paths: dict[str, Path] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Summarize CHARMM MM pretreat state before MLpot registration."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import validate_cluster_geometry
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        _read_charmm_box_sides_A,
        charmm_crystal_is_active,
        parse_cubic_box_side_from_charmm_restart,
        probe_charmm_cubic_box_side_A,
        resolve_charmm_cubic_box_side_A,
        resolve_mlpot_mic_box_side_A,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    r = np.asarray(positions, dtype=float)
    stats = validate_cluster_geometry(r, n_molecules=n_monomers)
    geom: dict[str, Any] = {
        "tag": tag,
        "n_monomers": int(n_monomers),
        "n_atoms": int(r.shape[0]),
        "span_x_Å": f"{stats['span_x']:.3f}",
        "span_y_Å": f"{stats['span_y']:.3f}",
        "span_z_Å": f"{stats['span_z']:.3f}",
    }
    if n_monomers > 1 and not np.isnan(stats.get("com_dist_min", np.nan)):
        geom["com_dist_min_Å"] = f"{stats['com_dist_min']:.3f}"
        geom["com_dist_max_Å"] = f"{stats['com_dist_max']:.3f}"

    mm_state: dict[str, Any] = {
        "coord_source": "in-memory CHARMM (pretreat; no restart reload)",
        "GRMS_kcal/mol/Å": f"{charmm_grms():.4f}",
        "coord_rows": int(r.shape[0]),
    }
    live = get_charmm_positions_array()
    if live is not None:
        mm_state["live_coord_rows"] = int(live.shape[0])
        if live.shape[0] != r.shape[0]:
            mm_state["coord_row_mismatch"] = True

    pbc: dict[str, Any] = {}
    warnings: list[str] = []
    if use_pbc:
        restart_path = Path(pretreat_restart) if pretreat_restart is not None else None
        restart_side = (
            parse_cubic_box_side_from_charmm_restart(restart_path)
            if restart_path is not None and restart_path.is_file()
            else None
        )
        lx, ly, lz = _read_charmm_box_sides_A()
        crystal_active = charmm_crystal_is_active()
        pbound_side, pbound_source = probe_charmm_cubic_box_side_A()
        restart_for_workflow = None if crystal_active else restart_path
        workflow_side, workflow_source = resolve_charmm_cubic_box_side_A(
            fallback_side_A=workflow_box_side_A,
            restart_path=restart_for_workflow,
        )
        ml_side, ml_source = resolve_mlpot_mic_box_side_A(
            fallback_side_A=workflow_box_side_A,
            restart_path=restart_path,
        )

        pbc = {
            "crystal_active": crystal_active,
            "pbound_Å": f"({lx:.3f}, {ly:.3f}, {lz:.3f})",
            "pbound_cubic_L_Å": (
                f"{pbound_side:.3f} ({pbound_source})"
                if pbound_side is not None and pbound_source == "pbound"
                else "inactive"
            ),
            "workflow_box_L_Å": f"{workflow_side:.3f} ({workflow_source})",
            "MLpot_MIC_sync_L_Å": f"{ml_side:.3f} ({ml_source})",
        }
        if restart_path is not None and restart_path.is_file():
            pbc["pretreat_restart"] = restart_path.name
        if restart_side is not None:
            pbc["restart_crystal_L_Å"] = f"{restart_side:.3f}"
        if workflow_box_side_A is not None:
            pbc["campaign_box_size_Å"] = f"{float(workflow_box_side_A):.3f}"
        if workflow_side > 0.0:
            rho = float(r.shape[0]) / float(workflow_side) ** 3
            pbc["number_density_atoms/Å³"] = f"{rho:.5f}"

        if workflow_source == "restart":
            warnings.append(
                "workflow box resolved from restart file (pbound not cubic/active); "
                "MLpot may log source=restart — prefer live pbound when IMAGE is active"
            )
        if ml_source == "restart" and crystal_active:
            warnings.append(
                "crystal/IMAGE appear active but MLpot MIC sync will read restart crystal"
            )
        if (
            restart_side is not None
            and pbound_side is not None
            and pbound_source == "pbound"
            and abs(restart_side - pbound_side) > max(1e-3, 1e-4 * pbound_side)
        ):
            warnings.append(
                f"restart crystal L={restart_side:.3f} Å ≠ live pbound L={pbound_side:.3f} Å"
            )
        if (
            workflow_box_side_A is not None
            and workflow_source == "pbound"
            and abs(workflow_side - float(workflow_box_side_A)) > max(1e-3, 1e-4 * workflow_side)
        ):
            warnings.append(
                f"campaign --box-size {float(workflow_box_side_A):.3f} Å "
                f"≠ live pbound {workflow_side:.3f} Å (NPT drift or pretreat resize)"
            )

    artifacts: dict[str, Any] = {}
    if paths:
        for key, label in (
            ("charmm_mm_heat_res", "heat.res"),
            ("charmm_mm_equi_res", "equi.res"),
            ("charmm_mm_prod_res", "prod.res"),
        ):
            candidate = paths.get(key)
            if candidate is not None and Path(candidate).is_file():
                artifacts[label] = Path(candidate).name

    sections: list[tuple[str, dict[str, Any]]] = [
        ("Geometry", geom),
        ("CHARMM MM state", mm_state),
    ]
    if pbc:
        sections.append(("PBC → MLpot handoff", pbc))
    if artifacts:
        sections.append(("Pretreat artifacts", artifacts))
    if warnings:
        sections.append(
            ("Warnings", {f"warn_{i + 1}": w for i, w in enumerate(warnings)})
        )
    return sections


def print_charmm_mm_pretreat_handoff_panel(
    positions: np.ndarray,
    *,
    n_monomers: int,
    tag: str,
    pretreat_restart: PathLike | None = None,
    workflow_box_side_A: float | None = None,
    use_pbc: bool = False,
    paths: dict[str, Path] | None = None,
    quiet: bool = False,
) -> None:
    """Rich dashboard after CHARMM MM pretreat, before MLpot registration."""
    from mmml.utils.rich_report import emit_dashboard

    sections = build_charmm_mm_pretreat_handoff_sections(
        positions,
        n_monomers=n_monomers,
        tag=tag,
        pretreat_restart=pretreat_restart,
        workflow_box_side_A=workflow_box_side_A,
        use_pbc=use_pbc,
        paths=paths,
    )
    emit_dashboard(
        "CHARMM MM pretreat → MLpot handoff",
        sections,
        border_style="green",
        quiet=quiet,
    )


def run_workflow(
    args: argparse.Namespace,
    *,
    phase: Phase,
    ensemble: Ensemble = "nve",
) -> int:
    if getattr(args, "mlpot_profile", False):
        import os
        os.environ["MMML_MLPOT_PROFILE"] = "1"
        os.environ["MMML_JAX_COMPILE_TIMERS"] = "1"
    setattr(args, "ensemble", ensemble)
    if phase == "minimize":
        setattr(args, "setup", getattr(args, "setup", None) or "pycharmm_minimize")
        from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
            run_staged_workflow,
        )

        return run_staged_workflow(args)
    if phase in ("dynamics", "full", "staged"):
        if phase == "full" and not getattr(args, "no_pre_minimize", False):
            if not getattr(args, "md_stages", None) and not getattr(args, "md_stage", None):
                setup = getattr(args, "setup", "") or ""
                if setup == "free_nvt":
                    setattr(args, "md_stages", "mini,heat")
                elif setup == "free_nve":
                    setattr(args, "md_stages", "mini,nve")
        from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
            run_staged_workflow,
        )

        return run_staged_workflow(args)
    raise ValueError(f"Unknown phase: {phase}")
