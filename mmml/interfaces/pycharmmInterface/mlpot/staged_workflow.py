"""Multi-stage MLpot MD: mini → heat → NVE → equi → production (+ PBC)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_charmm_output_from_args,
    apply_flat_bottom_from_args,
    assert_dynamics_ready,
    build_cluster_from_args_with_tag,
    charmm_grms,
    dynamics_nstep_from_ps,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_constrain_resids,
    resolve_dcd_nsavc,
    resolve_dynamics_print_kwargs,
    resolve_echeck_for_cluster,
    resolve_fix_resids,
    resolve_mini_nstep,
    resolve_md_stages,
    resolve_pbc_box_side,
    resolve_show_energy,
    resolve_test_first_config,
    resolve_use_pbc,
    setup_cons_fix_for_resids,
    timestep_ps_from_dt_fs,
    turn_off_cons_fix,
    validate_resids_for_cluster,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmTrajectoryFiles,
    MinimizeWithMlpotConfig,
    build_cpt_equilibration_dynamics,
    build_cpt_production_dynamics,
    build_heat_dynamics,
    build_nve_dynamics,
    minimize_with_mlpot,
    npt_restart_chain,
    production_restart_chain,
    run_dynamics_with_io,
)
from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
    assert_pre_min_bonded_geometry,
    maybe_run_bonded_mm_mini_after_stage,
    record_mm_baseline_strain,
    rewrite_dynamics_restart_from_current_state,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
    _charmm_pre_minimize_before_mlpot,
    _register_mlpot_context,
    sync_mlpot_pbc_cell_from_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    disable_charmm_domdec,
    get_charmm_positions_array,
    load_cluster_from_artifacts,
    save_cluster_topology_for_vmd,
    select_by_resids,
    sync_charmm_positions,
)

MdStage = Literal["mini", "heat", "nve", "equi", "prod"]

_STAGE_ORDER: tuple[MdStage, ...] = ("mini", "heat", "nve", "equi", "prod")


def _stage_ps(args: argparse.Namespace, stage: MdStage) -> float:
    dt_fs = float(getattr(args, "dt_fs", 0.25))
    if stage == "heat":
        return float(getattr(args, "ps_heat", 10.0))
    if stage == "equi":
        return float(getattr(args, "ps_equi", 50.0))
    if stage == "prod":
        return float(getattr(args, "ps_prod", None) or getattr(args, "ps", 100.0))
    if stage == "nve":
        return float(getattr(args, "ps_nve", None) or getattr(args, "ps", 50.0))
    return float(getattr(args, "ps", 1.0))


def _artifact_paths(out_dir: Path, tag: str) -> dict[str, Path]:
    return {
        "mini_crd": out_dir / f"mini_full_mlpot_{tag}.crd",
        "mini_psf": out_dir / f"mini_full_mlpot_{tag}.psf",
        "mini_dcd": out_dir / f"mini_full_mlpot_{tag}.dcd",
        "heat_res": out_dir / f"heat_{tag}.res",
        "heat_dcd": out_dir / f"heat_{tag}.dcd",
        "nve_res": out_dir / f"nve_{tag}.res",
        "nve_dcd": out_dir / f"nve_{tag}.dcd",
        "equi_res": out_dir / f"equi_{tag}.res",
        "equi_dcd": out_dir / f"equi_{tag}.dcd",
        "prod_res": out_dir / f"prod_{tag}.res",
        "prod_dcd": out_dir / f"prod_{tag}.dcd",
        "vmd_psf": out_dir / f"cluster_for_vmd_{tag}.psf",
    }


def _npt_cpt_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "thermostat": getattr(args, "npt_thermostat", "hoover"),
        "pref": float(getattr(args, "npt_pressure", 1.0)),
        "pgamma": float(getattr(args, "npt_pgamma", 5.0)),
    }


def _equi_restart_name(tag: str, n_equi_segments: int) -> str:
    if n_equi_segments > 1:
        return f"equi_{tag}.{n_equi_segments - 1}.res"
    return f"equi_{tag}.res"


def _prior_restart_for_stage(
    stage: MdStage,
    paths: dict[str, Path],
    *,
    restart_from: Path | None,
) -> Path | None:
    if restart_from is not None:
        return restart_from
    if stage == "heat":
        return None
    if stage == "nve":
        return paths["heat_res"] if paths["heat_res"].is_file() else None
    if stage == "equi":
        if paths["nve_res"].is_file():
            return paths["nve_res"]
        if paths["heat_res"].is_file():
            return paths["heat_res"]
        return None
    if stage == "prod":
        return paths["equi_res"] if paths["equi_res"].is_file() else None
    return None


def _io_for_stage(stage: MdStage, paths: dict[str, Path]) -> CharmmTrajectoryFiles:
    if stage == "heat":
        return CharmmTrajectoryFiles(
            restart_write=paths["heat_res"],
            trajectory=paths["heat_dcd"],
        )
    if stage == "nve":
        return CharmmTrajectoryFiles(
            restart_write=paths["nve_res"],
            trajectory=paths["nve_dcd"],
        )
    if stage == "equi":
        return CharmmTrajectoryFiles(
            restart_write=paths["equi_res"],
            trajectory=paths["equi_dcd"],
        )
    if stage == "prod":
        return CharmmTrajectoryFiles(
            restart_write=paths["prod_res"],
            trajectory=paths["prod_dcd"],
        )
    raise ValueError(f"no dynamics I/O for stage {stage!r}")


def _sync_mlpot_cell_before_npt(
    stage: MdStage,
    *,
    use_pbc: bool,
    pyCModel: Any,
    quiet: bool,
    restart_path: Path | None = None,
) -> None:
    if use_pbc and stage in ("equi", "prod"):
        sync_mlpot_pbc_cell_from_charmm(
            pyCModel,
            verbose=not quiet,
            restart_path=restart_path,
        )


def _build_stage_dynamics_kw(
    stage: MdStage,
    *,
    args: argparse.Namespace,
    timestep_ps: float,
    nstep: int,
    save_interval_ps: float,
    temp: float,
    echeck: float,
    dyn_print: dict[str, int],
    restart: bool,
    npt_include_firstt: bool = True,
    memory_handoff: bool = False,
) -> dict[str, Any]:
    duration_ps = nstep * timestep_ps
    effective_restart = restart and not memory_handoff
    if stage == "heat":
        kw = build_heat_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=duration_ps,
            save_interval_ps=save_interval_ps,
            temp=temp,
            echeck=echeck,
        )
    elif stage == "nve":
        kw = build_nve_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=duration_ps,
            save_interval_ps=save_interval_ps,
            restart=effective_restart,
            temp=temp,
            nprint=dyn_print["nprint"],
            iprfrq=dyn_print["iprfrq"],
            isvfrq=dyn_print["isvfrq"],
            echeck=echeck,
        )
    elif stage == "equi":
        kw = build_cpt_equilibration_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=duration_ps,
            save_interval_ps=save_interval_ps,
            temp=temp,
            restart=effective_restart,
            echeck=max(echeck, 500.0) if echeck > 0 else echeck,
            include_firstt=npt_include_firstt,
            **_npt_cpt_options(args),
        )
    elif stage == "prod":
        kw = build_cpt_production_dynamics(
            timestep_ps=timestep_ps,
            duration_ps=duration_ps,
            save_interval_ps=save_interval_ps,
            temp=temp,
            restart=effective_restart,
            echeck=max(echeck, 500.0) if echeck > 0 else echeck,
            **_npt_cpt_options(args),
        )
    else:
        raise ValueError(stage)
    kw["nprint"] = dyn_print["nprint"]
    kw["iprfrq"] = dyn_print["iprfrq"]
    kw["isvfrq"] = dyn_print["isvfrq"]
    kw["nstep"] = nstep
    if memory_handoff:
        kw["new"] = False
        kw["start"] = False
        kw["restart"] = False
    elif restart:
        kw["new"] = False
        kw["start"] = False
        kw["restart"] = True
    else:
        kw["new"] = True
        kw["start"] = True
    return kw


def _reset_stage_trajectory(path: Path | None) -> None:
    """Remove a prior partial DCD so a new stage run starts a fresh trajectory file."""
    if path is not None:
        Path(path).unlink(missing_ok=True)


def _seed_restart_for_memory_handoff(
    io: CharmmTrajectoryFiles,
    kw: dict[str, Any],
    *,
    stage: MdStage,
) -> Path:
    """Write in-memory state to ``restart_write`` and switch ``kw`` to ``restart -``."""
    if io.restart_write is None:
        raise RuntimeError(
            f"memory handoff for stage {stage!r} requires restart_write on I/O"
        )
    rewrite_dynamics_restart_from_current_state(io.restart_write)
    seed = Path(io.restart_write)
    io.restart_read = seed
    kw["new"] = False
    kw["start"] = False
    kw["restart"] = True
    if stage in ("equi", "prod"):
        kw.pop("firstt", None)
    return seed


def _load_or_build_cluster(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    if getattr(args, "skip_cluster_build", False) or getattr(args, "from_psf", None):
        return load_cluster_from_artifacts(args)
    return build_cluster_from_args_with_tag(args)


def run_staged_workflow(args: argparse.Namespace) -> int:
    stages = resolve_md_stages(args)
    if getattr(args, "no_pre_minimize", False):
        stages = [s for s in stages if s != "mini"]
    if not stages:
        raise ValueError("no MD stages selected")

    fix_resids = resolve_fix_resids(args)
    dynamics_constrain = resolve_constrain_resids(args)
    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_mol, tag = _load_or_build_cluster(args)
    validate_resids_for_cluster(fix_resids, n_mol)
    validate_resids_for_cluster(dynamics_constrain, n_mol)
    print_cluster_geometry_summary(r, n_mol)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_atoms = len(z)
    paths = _artifact_paths(out_dir, tag)

    use_pbc = resolve_use_pbc(args)
    box_side = resolve_pbc_box_side(args, r) if use_pbc else None
    if use_pbc and not args.quiet:
        print(f"PBC cubic box: {box_side:.3f} Å", flush=True)

    dt_fs = float(getattr(args, "dt_fs", 0.25))
    timestep_ps = timestep_ps_from_dt_fs(dt_fs)
    if getattr(args, "timestep_ps", None) is not None:
        timestep_ps = float(args.timestep_ps)
    temp = float(getattr(args, "temperature", getattr(args, "temp", 300.0)))
    mini_nprint = apply_charmm_output_from_args(args)
    show_energy = resolve_show_energy(args)
    echeck = resolve_echeck_for_cluster(args, n_atoms=n_atoms, n_monomers=n_mol)
    mini_nstep = resolve_mini_nstep(args, n_mol)
    overlap_cfg = resolve_dynamics_overlap_config(
        args,
        n_monomers=n_mol,
        use_pbc=use_pbc,
        fallback_box_side_A=box_side if use_pbc else None,
    )
    if overlap_cfg.enabled and not args.quiet:
        print(
            f"Dynamics overlap guard: action={overlap_cfg.action}, "
            f"min_distance={overlap_cfg.min_distance_A:.2f} Å, "
            f"check every {overlap_cfg.check_interval} steps"
            + (
                ", last-resort monomer separation on"
                if overlap_cfg.separate_on_rescue_fail
                else ""
            ),
            flush=True,
        )

    setup_charmm_environment(use_pbc=use_pbc, cubic_box_side_A=box_side)
    sync_charmm_positions(r)

    vmd_topo_psf = paths["vmd_psf"]
    if not getattr(args, "no_save_vmd_topology", False) and not getattr(
        args, "skip_cluster_build", False
    ):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]

    if "mini" in stages and not getattr(args, "skip_cluster_build", False):
        r = _charmm_pre_minimize_before_mlpot(
            args, nprint=mini_nprint, reference_positions=r
        )
        sync_charmm_positions(r)

    baseline = None
    if getattr(args, "bonded_mm_mini", False) and getattr(args, "charmm_pre_minimize", True):
        baseline = record_mm_baseline_strain(verbose=not args.quiet)
        assert_pre_min_bonded_geometry(args, baseline=baseline)

    ctx, pyCModel = _register_mlpot_context(
        z,
        r,
        ckpt,
        n_atoms,
        n_mol,
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_gpu_count=getattr(args, "ml_gpu_count", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        cubic_box_side_A=box_side if use_pbc else None,
        verbose=not args.quiet,
        args=args,
    )

    restart_from = (
        Path(args.restart_from).expanduser().resolve()
        if getattr(args, "restart_from", None)
        else None
    )
    last_traj: Path | None = None
    try:
        if "mini" in stages:
            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            save_mini = bool(getattr(args, "save", True))
            dcd_nsavc = resolve_dcd_nsavc(
                dcd_nsavc=args.dcd_nsavc, nstep=mini_nstep
            )
            if not args.quiet:
                print(
                    f"\nMLpot SD minimize: {mini_nstep} steps/pass, {n_atoms} atoms",
                    flush=True,
                )
            minimize_with_mlpot(
                MinimizeWithMlpotConfig(
                    fixed_ml_selection=fix_sel,
                    nstep=mini_nstep,
                    nprint=mini_nprint,
                    verbose=not args.quiet,
                    reference_positions=r,
                    pyCModel=pyCModel,
                    save=save_mini,
                    pdb_path=out_dir / f"mini_full_mlpot_{tag}.pdb" if save_mini else None,
                    crd_path=paths["mini_crd"] if save_mini else None,
                    psf_path=paths["mini_psf"] if save_mini else None,
                    energy_json_path=out_dir / f"mini_full_mlpot_{tag}_energy.json"
                    if save_mini
                    else None,
                    xyz_path=out_dir / f"mini_full_mlpot_{tag}.xyz" if save_mini else None,
                    dcd_path=paths["mini_dcd"] if save_mini else None,
                    dcd_nsavc=dcd_nsavc if save_mini else 0,
                    skip_if_crd_exists=bool(getattr(args, "skip_if_crd_exists", False)),
                    test_first=resolve_test_first_config(args),
                    show_energy=show_energy,
                )
            )
            sync_charmm_positions(get_charmm_positions_array())
            if not args.quiet:
                print(f"Post MLpot mini GRMS: {charmm_grms():.4f} kcal/mol/Å", flush=True)
            last_traj = paths["mini_dcd"] if paths["mini_dcd"].is_file() else None

        dyn_stages = [s for s in _STAGE_ORDER if s in stages and s != "mini"]
        if not dyn_stages:
            return 0

        if not use_pbc:
            apply_flat_bottom_from_args(args)

        assert_dynamics_ready(
            max_grms=float(getattr(args, "max_grms_before_dyn", 50.0)),
            abort=not getattr(args, "allow_high_grms", False),
        )

        if dynamics_constrain:
            setup_cons_fix_for_resids(dynamics_constrain)

        n_equi_segments = max(1, int(getattr(args, "n_equi_segments", 1)))
        n_prod_segments = max(1, int(getattr(args, "n_prod_segments", 1)))
        if "equi" in dyn_stages and n_equi_segments > 1:
            equi_idx = dyn_stages.index("equi")
            dyn_stages = dyn_stages[:equi_idx] + ["equi"] * n_equi_segments
        if "prod" in dyn_stages and n_prod_segments > 1:
            prod_idx = dyn_stages.index("prod")
            dyn_stages = dyn_stages[:prod_idx] + ["prod"] * n_prod_segments

        equi_restart_for_prod = _equi_restart_name(tag, n_equi_segments)
        prev_restart: Path | None = restart_from
        memory_handoff_next = False
        for stage in dyn_stages:
            if stage == "equi" and n_equi_segments > 1:
                initial = prev_restart or _prior_restart_for_stage(
                    "equi", paths, restart_from=None
                )
                seg_chain = npt_restart_chain(
                    out_dir,
                    n_segments=n_equi_segments,
                    prefix=f"equi_{tag}",
                    initial_restart=initial,
                )
                for seg_i, seg_io in enumerate(seg_chain):
                    seg_ps = _stage_ps(args, "equi") / n_equi_segments
                    nstep = dynamics_nstep_from_ps(seg_ps, dt_fs)
                    dcd_nsavc = resolve_dcd_nsavc(
                        dcd_nsavc=args.dcd_nsavc,
                        dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
                        timestep_ps=timestep_ps,
                        nstep=nstep,
                    )
                    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
                    save_interval_ps = timestep_ps * dcd_nsavc
                    use_memory = memory_handoff_next
                    if use_memory:
                        restart = False
                        rread = None
                    else:
                        rread = seg_io.restart_read
                        restart = rread is not None and Path(rread).is_file()
                    if not args.quiet:
                        print(
                            f"\nEQUI segment {seg_i + 1}/{n_equi_segments}: "
                            f"{nstep} steps @ {timestep_ps} ps"
                            + (" | memory handoff" if use_memory else ""),
                            flush=True,
                        )
                    restart_path = Path(rread) if restart and rread else None
                    kw = _build_stage_dynamics_kw(
                        "equi",
                        args=args,
                        timestep_ps=timestep_ps,
                        nstep=nstep,
                        save_interval_ps=save_interval_ps,
                        temp=temp,
                        echeck=echeck,
                        dyn_print=dyn_print,
                        restart=restart,
                        npt_include_firstt=(seg_i == 0),
                        memory_handoff=use_memory,
                    )
                    kw["nsavc"] = dcd_nsavc
                    if use_memory:
                        restart_path = _seed_restart_for_memory_handoff(seg_io, kw, stage="equi")
                    _sync_mlpot_cell_before_npt(
                        "equi",
                        use_pbc=use_pbc,
                        pyCModel=pyCModel,
                        quiet=bool(args.quiet),
                        restart_path=restart_path,
                    )
                    disable_charmm_domdec()
                    _reset_stage_trajectory(
                        Path(seg_io.trajectory) if seg_io.trajectory else None
                    )
                    run_dynamics_with_io(
                        kw,
                        seg_io,
                        overlap=overlap_cfg,
                        overlap_context=f"equi segment {seg_i + 1}/{n_equi_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                    )
                    memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                        ctx,
                        args,
                        stage="equi",
                        baseline=baseline,
                        restart_path=seg_io.restart_write,
                    )
                    prev_restart = seg_io.restart_write
                    last_traj = seg_io.trajectory
                continue

            if stage == "prod" and n_prod_segments > 1:
                seg_chain = production_restart_chain(
                    out_dir,
                    n_segments=n_prod_segments,
                    prefix=f"prod_{tag}",
                    equi_restart=equi_restart_for_prod,
                )
                for seg_i, seg_io in enumerate(seg_chain):
                    seg_ps = _stage_ps(args, "prod") / n_prod_segments
                    nstep = dynamics_nstep_from_ps(seg_ps, dt_fs)
                    dcd_nsavc = resolve_dcd_nsavc(
                        dcd_nsavc=args.dcd_nsavc,
                        dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
                        timestep_ps=timestep_ps,
                        nstep=nstep,
                    )
                    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
                    save_interval_ps = timestep_ps * dcd_nsavc
                    use_memory = memory_handoff_next
                    if use_memory:
                        restart = False
                        rread = None
                    else:
                        rread = seg_io.restart_read
                        restart = rread is not None and Path(rread).is_file()
                    kw = _build_stage_dynamics_kw(
                        "prod",
                        args=args,
                        timestep_ps=timestep_ps,
                        nstep=nstep,
                        save_interval_ps=save_interval_ps,
                        temp=temp,
                        echeck=echeck,
                        dyn_print=dyn_print,
                        restart=restart,
                        memory_handoff=use_memory,
                    )
                    kw["nsavc"] = dcd_nsavc
                    if not args.quiet:
                        print(
                            f"\nPROD segment {seg_i + 1}/{n_prod_segments}: "
                            f"{nstep} steps @ {timestep_ps} ps"
                            + (" | memory handoff" if use_memory else ""),
                            flush=True,
                        )
                    restart_path = Path(rread) if restart and rread else None
                    if use_memory:
                        seed = _seed_restart_for_memory_handoff(seg_io, kw, stage="prod")
                        restart_path = seed
                    _sync_mlpot_cell_before_npt(
                        "prod",
                        use_pbc=use_pbc,
                        pyCModel=pyCModel,
                        quiet=bool(args.quiet),
                        restart_path=restart_path,
                    )
                    disable_charmm_domdec()
                    _reset_stage_trajectory(
                        Path(seg_io.trajectory) if seg_io.trajectory else None
                    )
                    run_dynamics_with_io(
                        kw,
                        seg_io,
                        overlap=overlap_cfg,
                        overlap_context=f"prod segment {seg_i + 1}/{n_prod_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                    )
                    memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                        ctx,
                        args,
                        stage="prod",
                        baseline=baseline,
                        restart_path=seg_io.restart_write,
                    )
                    last_traj = seg_io.trajectory
                continue

            stage_ps = _stage_ps(args, stage)
            nstep = dynamics_nstep_from_ps(stage_ps, dt_fs)
            dcd_nsavc = resolve_dcd_nsavc(
                dcd_nsavc=args.dcd_nsavc,
                dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
                timestep_ps=timestep_ps,
                nstep=nstep,
            )
            dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
            save_interval_ps = timestep_ps * dcd_nsavc

            use_memory = memory_handoff_next
            if use_memory:
                restart = False
                rread = None
            else:
                rread = prev_restart or _prior_restart_for_stage(stage, paths, restart_from=None)
                restart = rread is not None and Path(rread).is_file()
            io = _io_for_stage(stage, paths)
            if restart and rread is not None:
                io.restart_read = Path(rread)

            if not args.quiet:
                print(
                    f"\n{stage.upper()}: {nstep} steps @ {timestep_ps} ps | "
                    f"restart={restart}"
                    + (" | memory handoff" if use_memory else "")
                    + f" | {format_resid_constraint_message(dynamics_constrain, context='cons_fix')}",
                    flush=True,
                )

            restart_path = Path(rread) if restart and rread else None

            kw = _build_stage_dynamics_kw(
                stage,
                args=args,
                timestep_ps=timestep_ps,
                nstep=nstep,
                save_interval_ps=save_interval_ps,
                temp=temp,
                echeck=echeck,
                dyn_print=dyn_print,
                restart=restart,
                memory_handoff=use_memory,
            )
            kw["nsavc"] = dcd_nsavc
            if use_memory:
                restart_path = _seed_restart_for_memory_handoff(io, kw, stage=stage)
            _sync_mlpot_cell_before_npt(
                stage,
                use_pbc=use_pbc,
                pyCModel=pyCModel,
                quiet=bool(args.quiet),
                restart_path=restart_path,
            )
            disable_charmm_domdec()
            _reset_stage_trajectory(
                Path(io.trajectory) if io.trajectory else None
            )
            run_dynamics_with_io(
                kw,
                io,
                overlap=overlap_cfg,
                overlap_context=stage.upper(),
                mlpot_ctx=ctx,
                rng_base=getattr(args, "seed", None),
            )
            memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                ctx,
                args,
                stage=stage,
                baseline=baseline,
                restart_path=io.restart_write,
            )
            prev_restart = io.restart_write
            last_traj = io.trajectory

    finally:
        if dynamics_constrain:
            turn_off_cons_fix()
        ctx.unset()

    from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import maybe_log_mlpot_profile

    maybe_log_mlpot_profile(quiet=bool(args.quiet))
    print(f"\nStaged workflow OK ({','.join(stages)}) -> {out_dir}")
    if last_traj is not None and last_traj.is_file():
        print_vmd_load_help(
            out_dir=out_dir,
            tag=tag,
            topology_psf=vmd_topo_psf,
            trajectory=last_traj,
            n_atoms=n_atoms,
            bondless_psf=paths["mini_psf"] if paths["mini_psf"].is_file() else None,
        )
    return 0
