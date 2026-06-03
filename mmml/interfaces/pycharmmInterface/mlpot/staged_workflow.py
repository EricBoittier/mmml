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
    resolve_heat_firstt_finalt,
    resolve_heat_ihtfrq,
    resolve_heat_thermostat,
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
    _valid_restart_file,
    build_cpt_equilibration_dynamics,
    build_cpt_production_dynamics,
    build_heat_dynamics,
    build_nvt_equilibration_dynamics,
    build_nvt_production_dynamics,
    build_nve_dynamics,
    minimize_with_mlpot,
    npt_restart_chain,
    production_restart_chain,
    run_dynamics_with_io,
)
from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    apply_comp_velocity_policy,
)
from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
    assert_pre_min_bonded_geometry,
    maybe_run_bonded_mm_mini_after_stage,
    record_mm_baseline_strain,
    rewrite_dynamics_restart_from_current_state,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    DynamicsOverlapConfig,
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
    _charmm_pre_minimize_before_mlpot,
    _register_mlpot_context,
    run_charmm_mm_pretreat_before_mlpot,
    sync_mlpot_pbc_cell_from_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    assert_mlpot_user_active,
    verify_mlpot_charmm_atom_consistency,
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
        "mini_charmm_dcd": out_dir / f"mini_charmm_mm_{tag}.dcd",
        "mini_dcd": out_dir / f"mini_full_mlpot_{tag}.dcd",
        "charmm_mm_heat_res": out_dir / f"charmm_mm_heat_{tag}.res",
        "charmm_mm_heat_dcd": out_dir / f"charmm_mm_heat_{tag}.dcd",
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
    use_pbc: bool = True,
    npt_include_firstt: bool = True,
    memory_handoff: bool = False,
) -> dict[str, Any]:
    duration_ps = nstep * timestep_ps
    effective_restart = restart and not memory_handoff
    if stage == "heat":
        heat_firstt, heat_finalt = resolve_heat_firstt_finalt(args, default_temp=temp)
        if resolve_heat_thermostat(args) == "hoover":
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                build_hoover_heat_dynamics,
            )

            kw = build_hoover_heat_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                firstt=heat_firstt,
                finalt=heat_finalt,
                echeck=echeck,
                use_pbc=use_pbc,
            )
        else:
            kw = build_heat_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                firstt=heat_firstt,
                finalt=heat_finalt,
                echeck=echeck,
                use_pbc=use_pbc,
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
        include_firstt = npt_include_firstt and not effective_restart
        if use_pbc:
            kw = build_cpt_equilibration_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=effective_restart,
                echeck=max(echeck, 500.0) if echeck > 0 else echeck,
                include_firstt=include_firstt,
                **_npt_cpt_options(args),
            )
        else:
            kw = build_nvt_equilibration_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=effective_restart,
                echeck=max(echeck, 500.0) if echeck > 0 else echeck,
                include_firstt=include_firstt,
            )
    elif stage == "prod":
        if use_pbc:
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
            kw = build_nvt_production_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                restart=effective_restart,
                echeck=max(echeck, 500.0) if echeck > 0 else echeck,
            )
    else:
        raise ValueError(stage)
    kw["nprint"] = dyn_print["nprint"]
    kw["iprfrq"] = dyn_print["iprfrq"]
    kw["isvfrq"] = dyn_print["isvfrq"]
    kw["nstep"] = nstep
    if stage == "heat" or (
        stage == "equi"
        and not use_pbc
        and resolve_heat_thermostat(args) == "scale"
        and int(kw.get("ihtfrq", 0)) > 0
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import apply_heat_ramp_frequencies

        apply_heat_ramp_frequencies(
            kw,
            nstep=nstep,
            ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep),
        )
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


def _configure_heat_dynamics_start(
    kw: dict[str, Any],
    io: CharmmTrajectoryFiles,
    *,
    coords_in_memory: bool,
    restart_from_file: bool,
    timestep_ps: float,
    use_pbc: bool,
    quiet: bool,
    heat_thermostat: str = "scale",
) -> None:
    """Ensure heat has Boltzmann velocities at ``FIRSTT`` (DCM2-style ``start``).

    ``RESTART`` without ``START`` skips the initial assignment; mini restart files
    often carry ~zero kinetic energy, so ``ihtfrq`` with ``iasvel=0`` leaves T≈0.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        assign_velocities_at_temperature,
    )

    firstt = float(kw.get("firstt", kw.get("finalt", 300.0)))
    kw["iasvel"] = 1
    hoover_heat = heat_thermostat == "hoover"
    if not hoover_heat:
        # Scale at IHTFRQ (CHARMM iasors=0); avoid Gaussian reassignment every ihtfrq
        # which spikes T and trips echeck on all-ML clusters (no SHAKE).
        kw["iasors"] = 0

    if coords_in_memory:
        io.restart_read = None
        kw["restart"] = False
        kw["new"] = False
        assign_velocities_at_temperature(
            firstt,
            timestep_ps=timestep_ps,
            restart_path=None,
            use_pbc=use_pbc,
        )
        kw["start"] = False
        if not quiet:
            if hoover_heat:
                print(
                    f"HEAT: Boltzmann velocities at FIRSTT={firstt:.1f} K "
                    "(in-memory coords after mini); Hoover NVT (no ihtfrq)",
                    flush=True,
                )
            else:
                print(
                    f"HEAT: Boltzmann velocities at FIRSTT={firstt:.1f} K "
                    "(in-memory coords after mini); ihtfrq scales (iasors=0)",
                    flush=True,
                )
        return

    if restart_from_file and io.restart_read is not None:
        restart_path = io.restart_read
        assign_velocities_at_temperature(
            firstt,
            timestep_ps=timestep_ps,
            restart_path=restart_path,
            use_pbc=use_pbc,
        )
        io.restart_read = None
        kw["restart"] = False
        kw["new"] = False
        kw["start"] = False
        if not quiet:
            msg = (
                f"HEAT: Boltzmann velocities at FIRSTT={firstt:.1f} K "
                f"(coords from {restart_path}); "
            )
            print(
                msg
                + (
                    "Hoover NVT (no ihtfrq)"
                    if hoover_heat
                    else "ihtfrq scales (iasors=0)"
                ),
                flush=True,
            )
        return

    kw["restart"] = False
    kw["new"] = False
    kw["start"] = True
    if not quiet:
        print(
            f"HEAT: dyna start FIRSTT={firstt:.1f} K (cold start); "
            + (
                "Hoover NVT (no ihtfrq)"
                if hoover_heat
                else "ihtfrq scales (iasors=0)"
            ),
            flush=True,
        )


def _overlap_for_stage(
    stage: MdStage,
    overlap_cfg: DynamicsOverlapConfig,
) -> DynamicsOverlapConfig | None:
    """Return overlap guard config for dynamics stages (including heat).

    Heat uses the same chunked overlap checks and per-chunk DCD writes as equi/prod.
    Prefer ``--heat-thermostat hoover`` (default in ``run_dcm9_stability.sh``):
    velocity-scaling ramps (``ihtfrq``) do not survive overlap restart chunks.
    """
    return overlap_cfg


def _reset_stage_trajectory(
    path: Path | None,
    *,
    rescue_old: bool = False,
) -> None:
    """Ensure a stage DCD write starts from an empty file.

    Default: remove any prior ``path`` (fresh trajectory for this stage).
    With ``rescue_old=True``, rename the old file to ``*.rescued.N.dcd`` instead.
    """
    if path is None:
        return

    dcd_path = Path(path)
    if not dcd_path.exists():
        return

    if rescue_old:
        for rescue_index in range(1, 10_000):
            rescue_path = dcd_path.with_name(
                f"{dcd_path.stem}.rescued.{rescue_index}{dcd_path.suffix}"
            )
            if not rescue_path.exists():
                dcd_path.replace(rescue_path)
                print(f"Rescued existing DCD: {dcd_path} -> {rescue_path}", flush=True)
                return
        raise RuntimeError(f"could not find an available rescue name for {dcd_path}")

    dcd_path.unlink(missing_ok=True)
    print(f"Removed prior DCD: {dcd_path}", flush=True)


def _reset_stage_restart(
    restart_path: Path | None,
    *,
    trajectory_path: Path | None = None,
) -> None:
    """Remove prior stage restart/scratch files before a fresh dynamics run."""
    if restart_path is None:
        return
    path = Path(restart_path)
    if path.is_file():
        path.unlink(missing_ok=True)
        print(f"Removed prior restart: {path}", flush=True)
    parent = path.parent
    stem = path.stem
    for slot in (f"{stem}.overlap_a.res", f"{stem}.overlap_b.res"):
        Path(parent / slot).unlink(missing_ok=True)
    if trajectory_path is not None:
        traj_stem = Path(trajectory_path).stem
        for chunk_dcd in parent.glob(f"{traj_stem}.chunk.*.dcd"):
            chunk_dcd.unlink(missing_ok=True)


def _validate_dyn_stage_completion(
    args: argparse.Namespace,
    *,
    stage: str,
    nstep: int,
    nsavc: int,
    io: CharmmTrajectoryFiles,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )

    restart_path = Path(io.restart_write) if io.restart_write else None
    dcd_path = Path(io.trajectory) if io.trajectory else None
    assert_stage_dynamics_completed(
        stage=stage,
        expected_nstep=nstep,
        nsavc=nsavc,
        dcd_path=dcd_path,
        restart_path=restart_path,
        allow_incomplete=bool(getattr(args, "allow_incomplete_dynamics", False)),
    )


def _trajectory_outputs(path: Path | None) -> list[Path]:
    """Existing non-empty DCD output for a stage."""
    if path is None:
        return []
    stage_path = Path(path)
    if stage_path.is_file() and stage_path.stat().st_size > 0:
        return [stage_path]
    return []


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


def _can_seed_stage_from_memory(
    rread: Path | None,
    *,
    prev_restart: Path | None,
    prev_restart_is_current_state: bool,
) -> bool:
    """True when an invalid prior-stage restart can be replaced from live CHARMM state."""
    return (
        rread is not None
        and prev_restart is not None
        and prev_restart_is_current_state
        and Path(rread) == Path(prev_restart)
        and Path(rread).is_file()
        and _valid_restart_file(rread) is None
    )


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
    if overlap_cfg.intra_enabled and not args.quiet:
        print(
            f"Dynamics intra-monomer guard: action={overlap_cfg.action}, "
            f"min_distance={overlap_cfg.intra_min_distance_A:.2f} Å, "
            f"exclude_1_3={overlap_cfg.intra_exclude_1_3}, "
            f"check every {overlap_cfg.check_interval} steps",
            flush=True,
        )

    setup_charmm_environment(use_pbc=use_pbc, cubic_box_side_A=box_side)
    sync_charmm_positions(r)

    vmd_topo_psf = paths["vmd_psf"]
    if getattr(args, "skip_cluster_build", False) and getattr(args, "from_psf", None):
        vmd_topo_psf = Path(args.from_psf).expanduser().resolve()
    if not getattr(args, "no_save_vmd_topology", False) and not getattr(
        args, "skip_cluster_build", False
    ):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]
    recovery_topology_psf = vmd_topo_psf if Path(vmd_topo_psf).is_file() else None

    pretreat_mm = bool(getattr(args, "charmm_mm_pretreat", False))
    if pretreat_mm and not getattr(args, "skip_cluster_build", False):
        r = run_charmm_mm_pretreat_before_mlpot(
            args,
            paths=paths,
            timestep_ps=timestep_ps,
            use_pbc=use_pbc,
            temp=temp,
            echeck=echeck,
            mini_nprint=mini_nprint,
            reference_positions=r,
        )
        sync_charmm_positions(r)
    elif "mini" in stages and not getattr(args, "skip_cluster_build", False):
        save_mini = bool(getattr(args, "save", True))
        mini_dcd_nsavc = resolve_dcd_nsavc(
            dcd_nsavc=args.dcd_nsavc, nstep=mini_nstep
        )
        r = _charmm_pre_minimize_before_mlpot(
            args,
            nprint=mini_nprint,
            reference_positions=r,
            dcd_path=paths["mini_charmm_dcd"] if save_mini else None,
            dcd_nsavc=mini_dcd_nsavc if save_mini else 0,
        )
        sync_charmm_positions(r)

    if not use_pbc:
        # Install MMFP once after Packmol / CHARMM pretreat / pre-MLpot mini so
        # droff tuning and coor orient are not repeated (stacked walls / COM shifts).
        apply_flat_bottom_from_args(args)
        r = get_charmm_positions_array()

    baseline = None
    if (
        getattr(args, "bonded_mm_mini", False)
        and getattr(args, "charmm_pre_minimize", True)
        and not pretreat_mm
    ):
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
    last_restart_path: Path | None = None
    try:
        if "mini" in stages:
            fix_sel = select_by_resids(fix_resids) if fix_resids else None
            save_mini = bool(getattr(args, "save", True))
            mini_dcd_nsavc = resolve_dcd_nsavc(
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
                    dcd_nsavc=mini_dcd_nsavc if save_mini else 0,
                    skip_if_crd_exists=bool(getattr(args, "skip_if_crd_exists", False)),
                    test_first=resolve_test_first_config(args),
                    show_energy=show_energy,
                )
            )
            sync_charmm_positions(get_charmm_positions_array())
            if not args.quiet:
                print(f"Post MLpot mini GRMS: {charmm_grms():.4f} kcal/mol/Å", flush=True)
            mini_trajectories = _trajectory_outputs(paths["mini_charmm_dcd"])
            mini_trajectories.extend(_trajectory_outputs(paths["mini_dcd"]))
            last_traj = mini_trajectories[-1] if mini_trajectories else None
            maybe_run_bonded_mm_mini_after_stage(
                ctx,
                args,
                stage="mini",
                baseline=baseline,
                restart_path=paths["mini_crd"],
                topology_psf=recovery_topology_psf,
            )

        dyn_stages = [s for s in _STAGE_ORDER if s in stages and s != "mini"]
        if not dyn_stages:
            return 0

        assert_mlpot_user_active(ctx, context="staged dynamics", quiet=bool(args.quiet))
        assert_dynamics_ready(
            max_grms=float(getattr(args, "max_grms_before_dyn", 50.0)),
            abort=not getattr(args, "allow_high_grms", False),
            require_mlpot_user=True,
        )
        verify_mlpot_charmm_atom_consistency(
            ctx,
            expected_z=z,
            context="staged dynamics",
            quiet=bool(args.quiet),
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
        prev_restart_is_current_state = False
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
                        if _can_seed_stage_from_memory(
                            Path(rread) if rread is not None else None,
                            prev_restart=prev_restart,
                            prev_restart_is_current_state=prev_restart_is_current_state,
                        ):
                            use_memory = True
                            restart = False
                            rread = None
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
                        use_pbc=use_pbc,
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
                        Path(seg_io.trajectory) if seg_io.trajectory else None,
                        rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
                    )
                    assert_mlpot_user_active(
                        ctx,
                        context=f"equi segment {seg_i + 1}/{n_equi_segments}",
                        quiet=bool(args.quiet),
                    )
                    run_dynamics_with_io(
                        kw,
                        seg_io,
                        overlap=_overlap_for_stage("equi", overlap_cfg),
                        overlap_context=f"equi segment {seg_i + 1}/{n_equi_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                    )
                    _validate_dyn_stage_completion(
                        args,
                        stage="equi",
                        nstep=nstep,
                        nsavc=dcd_nsavc,
                        io=seg_io,
                    )
                    memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                        ctx,
                        args,
                        stage="equi",
                        baseline=baseline,
                        restart_path=seg_io.restart_write,
                        topology_psf=recovery_topology_psf,
                    )
                    prev_restart = seg_io.restart_write
                    prev_restart_is_current_state = True
                    last_restart_path = prev_restart
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
                        if _can_seed_stage_from_memory(
                            Path(rread) if rread is not None else None,
                            prev_restart=prev_restart,
                            prev_restart_is_current_state=prev_restart_is_current_state,
                        ):
                            use_memory = True
                            restart = False
                            rread = None
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
                        use_pbc=use_pbc,
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
                        Path(seg_io.trajectory) if seg_io.trajectory else None,
                        rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
                    )
                    assert_mlpot_user_active(
                        ctx,
                        context=f"prod segment {seg_i + 1}/{n_prod_segments}",
                        quiet=bool(args.quiet),
                    )
                    run_dynamics_with_io(
                        kw,
                        seg_io,
                        overlap=_overlap_for_stage("prod", overlap_cfg),
                        overlap_context=f"prod segment {seg_i + 1}/{n_prod_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                    )
                    _validate_dyn_stage_completion(
                        args,
                        stage="prod",
                        nstep=nstep,
                        nsavc=dcd_nsavc,
                        io=seg_io,
                    )
                    memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                        ctx,
                        args,
                        stage="prod",
                        baseline=baseline,
                        restart_path=seg_io.restart_write,
                        topology_psf=recovery_topology_psf,
                    )
                    prev_restart = seg_io.restart_write
                    prev_restart_is_current_state = True
                    last_restart_path = prev_restart
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
                if _can_seed_stage_from_memory(
                    Path(rread) if rread is not None else None,
                    prev_restart=prev_restart,
                    prev_restart_is_current_state=prev_restart_is_current_state,
                ):
                    use_memory = True
                    restart = False
                    rread = None
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
                use_pbc=use_pbc,
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
            _reset_stage_restart(
                Path(io.restart_write) if io.restart_write else None,
                trajectory_path=Path(io.trajectory) if io.trajectory else None,
            )
            _reset_stage_trajectory(
                Path(io.trajectory) if io.trajectory else None,
                rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
            )
            assert_mlpot_user_active(
                ctx,
                context=stage.upper(),
                quiet=bool(args.quiet),
            )
            apply_comp_velocity_policy(stage, kw, args)
            if stage == "heat":
                heat_thermostat = resolve_heat_thermostat(args)
                overlap_for_stage = _overlap_for_stage(stage, overlap_cfg)
                if (
                    overlap_for_stage is not None
                    and overlap_for_stage.enabled
                    and heat_thermostat == "scale"
                    and not args.quiet
                ):
                    print(
                        "HEAT overlap guard: chunked restarts may disrupt ihtfrq "
                        "ramps; prefer --heat-thermostat hoover (default).",
                        flush=True,
                    )
                _configure_heat_dynamics_start(
                    kw,
                    io,
                    coords_in_memory=use_memory or prev_restart_is_current_state,
                    restart_from_file=restart and io.restart_read is not None,
                    timestep_ps=timestep_ps,
                    use_pbc=use_pbc,
                    quiet=bool(args.quiet),
                    heat_thermostat=heat_thermostat,
                )
                if not args.quiet:
                    if heat_thermostat == "hoover":
                        print(
                            f"HEAT Hoover: {kw.get('firstt')} -> {kw.get('finalt')} K "
                            f"over {stage_ps} ps | hoover reft={kw.get('hoover reft')} K "
                            f"tmass={kw.get('tmass')} | ihtfrq=0",
                            flush=True,
                        )
                    else:
                        print(
                            f"HEAT ramp: {kw.get('firstt')} -> {kw.get('finalt')} K "
                            f"over {stage_ps} ps | ihtfrq={kw.get('ihtfrq')} "
                            f"TEMINC={float(kw.get('TEMINC', 0)):.4g} K | "
                            "iasors=0 (scale)",
                            flush=True,
                        )
            run_dynamics_with_io(
                kw,
                io,
                overlap=_overlap_for_stage(stage, overlap_cfg),
                overlap_context=stage.upper(),
                mlpot_ctx=ctx,
                rng_base=getattr(args, "seed", None),
            )
            _validate_dyn_stage_completion(
                args,
                stage=stage,
                nstep=nstep,
                nsavc=dcd_nsavc,
                io=io,
            )
            memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                ctx,
                args,
                stage=stage,
                baseline=baseline,
                restart_path=io.restart_write,
                topology_psf=recovery_topology_psf,
            )
            prev_restart = io.restart_write
            prev_restart_is_current_state = True
            last_restart_path = prev_restart
            last_traj = io.trajectory

    finally:
        if dynamics_constrain:
            turn_off_cons_fix()
        ctx.unset()

    from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import maybe_log_mlpot_profile

    maybe_log_mlpot_profile(quiet=bool(args.quiet))
    from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
        maybe_save_run_state_from_workflow,
    )

    maybe_save_run_state_from_workflow(
        args,
        positions=get_charmm_positions_array(),
        atomic_numbers=z,
        out_dir=out_dir,
        tag=tag,
        stages_completed=list(stages),
        last_restart=last_restart_path,
        last_trajectory=last_traj,
    )
    print(f"\nStaged workflow OK ({','.join(stages)}) -> {out_dir}")
    trajectory_outputs = _trajectory_outputs(paths["mini_charmm_dcd"])
    trajectory_outputs.extend(_trajectory_outputs(paths["mini_dcd"]))
    if last_traj is not None and last_traj not in trajectory_outputs:
        trajectory_outputs.extend(_trajectory_outputs(last_traj))
    if trajectory_outputs:
        print_vmd_load_help(
            out_dir=out_dir,
            tag=tag,
            topology_psf=vmd_topo_psf,
            trajectory=trajectory_outputs,
            n_atoms=n_atoms,
            bondless_psf=paths["mini_psf"] if paths["mini_psf"].is_file() else None,
        )
    return 0
