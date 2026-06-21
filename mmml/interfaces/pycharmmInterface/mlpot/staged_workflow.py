"""Multi-stage MLpot MD: mini → heat → NVE → equi → production (+ PBC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_charmm_output_from_args,
    apply_flat_bottom_from_args,
    assert_dynamics_ready,
    build_cluster_from_args_with_tag,
    charmm_grms,
    composition_tag,
    dynamics_nstep_from_ps,
    format_resid_constraint_message,
    print_cluster_geometry_summary,
    print_vmd_load_help,
    resolve_checkpoint,
    resolve_constrain_resids,
    resolve_dcd_nsavc_for_args,
    resolve_dynamics_print_kwargs,
    resolve_heat_firstt_finalt,
    resolve_heat_hoover_tmass,
    resolve_heat_ihtfrq,
    resolve_heat_thermostat,
    resolve_nve_boltzmann_temp,
    resolve_echeck_for_cluster,
    resolve_fix_resids,
    resolve_max_grms_before_dyn,
    resolve_mini_nstep,
    refresh_mlpot_energy_and_grms,
    resolve_md_stages,
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
    overlap_run_state_kwargs_from_args,
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
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    DynamicsOverlapConfig,
    attach_prior_segment_restart,
    augment_overlap_config_for_rescue,
    check_dynamics_overlap,
    overlap_config_for_stage,
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
    apply_comp_velocity_policy,
)
from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
    assert_pre_min_bonded_geometry,
    ensure_segment_restart_checkpoint,
    maybe_run_bonded_mm_mini_after_stage,
    record_mm_baseline_strain,
    rewrite_dynamics_restart_from_current_state,
)
from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
    attach_geometry_checkpoints_to_overlap,
    discover_resume_restart,
    write_geometry_baseline_restart,
)
from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
    ensure_charmm_crystal_for_cpt,
    setup_charmm_environment,
)
from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import (
    _charmm_pre_minimize_before_mlpot,
    _register_mlpot_context,
    run_charmm_mm_pretreat_before_mlpot,
    sync_mlpot_pbc_cell_from_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
    BONDED_MM_AFTER_HEAT,
    BONDED_MM_AFTER_MINI,
    CHARMM_MM_PRE,
    MLPOT_MMML,
    MinimizeArtifactRegistry,
    PACKMOL_CLUSTER,
    legacy_mlpot_mini_paths,
    mirror_legacy_mlpot_files,
    save_snapshot_from_charmm,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    assert_mlpot_user_active,
    verify_mlpot_charmm_atom_consistency,
    ensure_domdec_off_for_mlpot_energy,
    get_charmm_positions_array,
    load_cluster_from_artifacts,
    save_cluster_topology_for_vmd,
    select_by_resids,
    sync_charmm_positions,
)

MdStage = Literal["mini", "heat", "nve", "equi", "prod"]

_STAGE_ORDER: tuple[MdStage, ...] = ("mini", "heat", "nve", "equi", "prod")


def _stage_ps(args: argparse.Namespace, stage: MdStage) -> float:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_stage_ps

    return resolve_stage_ps(args, stage)


def _artifact_paths(out_dir: Path, tag: str) -> dict[str, Path]:
    from mmml.interfaces.pycharmmInterface.mlpot.minimize_artifacts import (
        BONDED_MM_AFTER_HEAT,
        BONDED_MM_AFTER_MINI,
        CHARMM_MM_PRE,
        MLPOT_MMML,
        legacy_charmm_mm_dcd,
        legacy_mlpot_mini_paths,
        snapshot_file_paths,
    )

    pretreat_dir = out_dir / "pretreat"
    legacy = legacy_mlpot_mini_paths(out_dir, tag)
    mm = snapshot_file_paths(pretreat_dir, CHARMM_MM_PRE, tag)
    mmml = snapshot_file_paths(out_dir, MLPOT_MMML, tag)
    bonded_mini = snapshot_file_paths(out_dir, BONDED_MM_AFTER_MINI, tag)
    bonded_heat = snapshot_file_paths(out_dir, BONDED_MM_AFTER_HEAT, tag)
    return {
        **legacy,
        "mini_crd": legacy["mini_crd"],
        "mini_psf": legacy["mini_psf"],
        "mini_pdb": legacy["mini_pdb"],
        "mini_charmm_dcd": legacy_charmm_mm_dcd(pretreat_dir, tag),
        "mini_dcd": legacy["mini_dcd"],
        "charmm_mm_crd": mm["crd"],
        "charmm_mm_pdb": mm["pdb"],
        "charmm_mm_psf": mm["psf"],
        "charmm_mm_energy_json": mm["energy_json"],
        "mlpot_mmml_crd": mmml["crd"],
        "mlpot_mmml_pdb": mmml["pdb"],
        "mlpot_mmml_psf": mmml["psf"],
        "mlpot_mmml_dcd": mmml["dcd"],
        "mlpot_mmml_xyz": mmml["xyz"],
        "mlpot_mmml_energy_json": mmml["energy_json"],
        "bonded_mm_after_mini_crd": bonded_mini["crd"],
        "bonded_mm_after_mini_pdb": bonded_mini["pdb"],
        "bonded_mm_after_heat_crd": bonded_heat["crd"],
        "bonded_mm_after_heat_pdb": bonded_heat["pdb"],
        "charmm_mm_heat_res": pretreat_dir / f"charmm_mm_heat_{tag}.res",
        "charmm_mm_heat_dcd": pretreat_dir / f"charmm_mm_heat_{tag}.dcd",
        "charmm_mm_equi_res": pretreat_dir / f"charmm_mm_equi_{tag}.res",
        "charmm_mm_equi_dcd": pretreat_dir / f"charmm_mm_equi_{tag}.dcd",
        "charmm_mm_prod_res": pretreat_dir / f"charmm_mm_prod_{tag}.res",
        "charmm_mm_prod_dcd": pretreat_dir / f"charmm_mm_prod_{tag}.dcd",
        "geometry_baseline_res": out_dir / f"geometry_baseline_{tag}.res",
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


def _heat_restart_path(paths: dict[str, Path], tag: str, n_heat_segments: int) -> Path:
    if n_heat_segments > 1:
        return paths["heat_res"].parent / f"heat_{tag}.{n_heat_segments - 1}.res"
    return paths["heat_res"]


def _prior_restart_for_stage(
    stage: MdStage,
    paths: dict[str, Path],
    *,
    restart_from: Path | None,
    tag: str | None = None,
    n_heat_segments: int = 1,
) -> Path | None:
    if stage == "heat":
        from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
            is_pretreat_mm_restart_path,
        )

        baseline = paths.get("geometry_baseline_res")
        if baseline is not None and Path(baseline).is_file():
            return Path(baseline)
        if restart_from is not None and not is_pretreat_mm_restart_path(restart_from):
            return restart_from
        return None
    if restart_from is not None:
        return restart_from
    if stage == "nve":
        heat_restart = _heat_restart_path(paths, tag or "", n_heat_segments)
        if heat_restart.is_file():
            return heat_restart
        if paths["heat_res"].is_file():
            return paths["heat_res"]
        return None
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
    mlpot_pbc: bool,
    pyCModel: Any,
    quiet: bool,
    restart_path: Path | None = None,
) -> None:
    if mlpot_pbc and stage in ("equi", "prod"):
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
        # ML USER-only heat (no SHAKE) often exceeds tight echeck before Hoover equilibrates.
        if getattr(args, "no_echeck_heat", False) or getattr(args, "no_echeck", False):
            heat_echeck = -1.0
        else:
            heat_echeck = max(echeck, 5000.0) if echeck > 0 else echeck
        if resolve_heat_thermostat(args) == "hoover":
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                build_hoover_heat_dynamics,
                compute_cpt_piston_masses,
            )

            tmass = None
            if use_pbc:
                _, psf_tmass = compute_cpt_piston_masses()
                tmass = resolve_heat_hoover_tmass(args, psf_tmass=psf_tmass)
            kw = build_hoover_heat_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                firstt=heat_firstt,
                finalt=heat_finalt,
                echeck=heat_echeck,
                use_pbc=use_pbc,
                ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep),
                tmass=tmass,
            )
        else:
            kw = build_heat_dynamics(
                timestep_ps=timestep_ps,
                duration_ps=duration_ps,
                save_interval_ps=save_interval_ps,
                temp=temp,
                firstt=heat_firstt,
                finalt=heat_finalt,
                echeck=heat_echeck,
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
            use_pbc=use_pbc,
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
    if stage == "heat":
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
            apply_heat_ramp_frequencies,
        )

        ihtfrq = resolve_heat_ihtfrq(args, nstep=nstep)
        if resolve_heat_thermostat(args) == "scale" or not kw.get("cpt"):
            # Scale heat and vacuum Hoover fallback (no CPT): ihtfrq velocity ramp.
            apply_heat_ramp_frequencies(kw, nstep=nstep, ihtfrq=ihtfrq)
        else:
            # PBC Hoover CPT: thermostat via hoover reft; disable IHTFRQ ramp.
            kw["ihtfrq"] = 0
            kw.pop("TEMINC", None)
    elif (
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

    In-place resume (``restart_read == restart_write``) uses ``dyna restart`` so
    the step counter and thermostat state continue from the checkpoint.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _valid_restart_file,
        assign_velocities_at_temperature,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_last_step,
    )

    firstt = float(kw.get("firstt", kw.get("finalt", 300.0)))
    kw["iasvel"] = 1
    hoover_cpt_heat = heat_thermostat == "hoover" and bool(kw.get("cpt"))
    if not hoover_cpt_heat:
        # Scale at IHTFRQ (CHARMM iasors=0); avoid Gaussian reassignment every ihtfrq
        # which spikes T and trips echeck on all-ML clusters (no SHAKE).
        kw["iasors"] = 0

    if coords_in_memory:
        io.restart_read = None
        kw["restart"] = False
        kw["new"] = False
        if hoover_cpt_heat:
            assign_velocities_at_temperature(
                firstt,
                timestep_ps=timestep_ps,
                restart_path=None,
                use_pbc=use_pbc,
            )
            # Velocities already drawn at FIRSTT; iasvel=1 on the main dyna would
            # re-assign at TBATH/FINALT (CHARMM) and spike T on segment ≥1 handoff.
            kw["iasvel"] = 0
            kw["start"] = False
        else:
            # Single dyna: Boltzmann at FIRSTT (start=True) then ihtfrq scaling.
            # Avoid a separate nstep=0 assign — it triggers a second dynamc/lambdata_init
            # that can segfault on BLOCK builds after the main heat segment starts.
            kw["iasvel"] = 1
            kw["iasors"] = 0
            kw["start"] = True
        if not quiet:
            if hoover_cpt_heat:
                print(
                    f"HEAT: Boltzmann velocities at FIRSTT={firstt:.1f} K "
                    "(in-memory coords after mini); Hoover CPT NVT (no ihtfrq); "
                    "start=False (no COMP velocity assign)",
                    flush=True,
                )
            else:
                print(
                    f"HEAT: dyna start FIRSTT={firstt:.1f} K "
                    "(in-memory coords after mini); ihtfrq scales (iasors=0)",
                    flush=True,
                )
        return

    if (
        restart_from_file
        and not coords_in_memory
        and _heat_in_place_restart(io)
        and _valid_restart_file(io.restart_read) is not None
    ):
        last_step = read_restart_last_step(Path(io.restart_read))
        if last_step is not None and last_step > 0:
            kw["restart"] = True
            kw["new"] = False
            kw["start"] = False
            kw["iasvel"] = 1
            if not hoover_cpt_heat:
                kw["iasors"] = 0
            if not quiet:
                print(
                    f"HEAT: dyna restart from {io.restart_read} "
                    f"(step {last_step}; in-place resume)",
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
        kw.pop("iunrea", None)
        kw["iunrea"] = -1
        if hoover_cpt_heat:
            kw["iasvel"] = 0
            kw["start"] = False
        else:
            kw["start"] = False
        if not quiet:
            msg = (
                f"HEAT: Boltzmann velocities at FIRSTT={firstt:.1f} K "
                f"(coords from {restart_path}); "
            )
            print(
                msg
                + (
                    "Hoover CPT NVT (no ihtfrq); start=False (no COMP velocity assign)"
                    if hoover_cpt_heat
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
                "Hoover CPT NVT (no ihtfrq)"
                if hoover_cpt_heat
                else "ihtfrq scales (iasors=0)"
            ),
            flush=True,
        )


def _configure_nve_dynamics_start(
    kw: dict[str, Any],
    io: CharmmTrajectoryFiles,
    *,
    coords_in_memory: bool,
    restart_from_file: bool,
    timestep_ps: float,
    use_pbc: bool,
    quiet: bool,
    temp: float,
) -> None:
    """One-shot Boltzmann draw at ``temp``, then microcanonical ``dyna`` (no START).

    After MLpot mini, coordinates are in memory but the saved CRD is not a CHARMM
    restart (memory handoff).  Omit START and keep ``iasvel=1`` so CHARMM cannot
    reuse comparison coordinates as velocities if START lingers from the assign
    call.  Mirrors the heat-stage ``assign_velocities_at_temperature`` pattern.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        assign_velocities_at_temperature,
    )

    target_t = float(temp)
    for key in ("iasors", "iscale", "iscvel", "ichecw", "firstt", "finalt", "tbath", "tstruct"):
        kw.pop(key, None)
    kw["iasvel"] = 1
    kw["ihtfrq"] = 0

    if coords_in_memory:
        io.restart_read = None
        kw["restart"] = False
        kw["new"] = False
        assign_velocities_at_temperature(
            target_t,
            timestep_ps=timestep_ps,
            restart_path=None,
            use_pbc=use_pbc,
        )
        kw["start"] = False
        # Match HEAT: continue from in-memory state. Do not READYN the scratch
        # restart written after the 1-step Boltzmann draw — CHARMM often leaves a
        # coordinate-history REST (EOF on READYN) while coords/vel are valid in RAM.
        if not quiet:
            print(
                f"NVE: Boltzmann velocities at {target_t:.1f} K "
                "(in-memory coords after mini); start omitted, iasvel=1",
                flush=True,
            )
        return

    if restart_from_file and io.restart_read is not None:
        restart_path = io.restart_read
        assign_velocities_at_temperature(
            target_t,
            timestep_ps=timestep_ps,
            restart_path=restart_path,
            use_pbc=use_pbc,
        )
        io.restart_read = None
        kw["restart"] = False
        kw["new"] = False
        kw["start"] = False
        if not quiet:
            print(
                f"NVE: Boltzmann velocities at {target_t:.1f} K "
                f"(coords from {restart_path}); start omitted, iasvel=1",
                flush=True,
            )
        return

    if not quiet:
        print(
            f"NVE: continuing velocities from restart "
            f"(start omitted, T target {target_t:.1f} K)",
            flush=True,
        )


def _overlap_for_stage(
    stage: MdStage,
    overlap_cfg: DynamicsOverlapConfig,
    *,
    ctx: Any = None,
    args: argparse.Namespace | None = None,
    topology_psf: Path | None = None,
    mini_registry: MinimizeArtifactRegistry | None = None,
) -> DynamicsOverlapConfig | None:
    """Return overlap guard config for dynamics stages (including heat).

    Heat uses the same chunked overlap checks and per-chunk DCD writes as equi/prod.
    Prefer ``--heat-thermostat hoover`` (default in ``run_dcm9_stability.sh``):
    velocity-scaling ramps (``ihtfrq``) do not survive overlap restart chunks.
    """
    if ctx is not None and args is not None:
        return augment_overlap_config_for_rescue(
            overlap_cfg,
            ctx=ctx,
            args=args,
            topology_psf=topology_psf,
            artifact_registry=mini_registry,
        )
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


def _heat_in_place_restart(io: CharmmTrajectoryFiles) -> bool:
    """True when heat reads and writes the same ``.res`` (resume interrupted heat)."""
    if io.restart_read is None or io.restart_write is None:
        return False
    return Path(io.restart_read).resolve() == Path(io.restart_write).resolve()


def _reset_stage_restart(
    restart_path: Path | None,
    *,
    trajectory_path: Path | None = None,
    restart_read: Path | None = None,
) -> None:
    """Remove prior stage restart/scratch files before a fresh dynamics run."""
    if restart_path is None:
        return
    path = Path(restart_path)
    if path.name.startswith("geometry_baseline_") and path.suffix == ".res":
        print(f"Keeping geometry baseline restart: {path}", flush=True)
        return
    preserve_main = (
        restart_read is not None
        and path.is_file()
        and path.resolve() == Path(restart_read).resolve()
    )
    if path.is_file() and not preserve_main:
        path.unlink(missing_ok=True)
        print(f"Removed prior restart: {path}", flush=True)
    elif preserve_main:
        print(f"Keeping in-place restart for resume: {path}", flush=True)
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
    segment_note: str | None = None,
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
        segment_note=segment_note,
    )


def _trajectory_outputs(path: Path | None) -> list[Path]:
    """Existing non-empty DCD output for a stage (including overlap chunk files)."""
    if path is None:
        return []
    stage_path = Path(path)
    outputs: list[Path] = []
    if stage_path.is_file() and stage_path.stat().st_size > 0:
        outputs.append(stage_path)
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        overlap_chunk_dcd_paths,
    )

    for chunk_path in overlap_chunk_dcd_paths(stage_path):
        if chunk_path.is_file() and chunk_path.stat().st_size > 0:
            outputs.append(chunk_path)
    return outputs


def _should_seed_heat_prior_restart(
    *,
    seg_i: int,
    prev_restart_is_current_state: bool,
    use_memory: bool,
    memory_handoff_next: bool,
) -> bool:
    """True when heat starts from in-memory coords and needs a fly-off checkpoint."""
    if seg_i == 0 and prev_restart_is_current_state:
        return True
    return bool(use_memory and (seg_i == 0 or memory_handoff_next))


def _overlap_extent_prior_restart(
    paths: dict[str, Path],
    prev_restart: Path | None,
) -> Path | None:
    """Best on-disk checkpoint for extent fly-off (post-mini baseline wins)."""
    baseline = paths.get("geometry_baseline_res")
    if baseline is not None and Path(baseline).is_file():
        return Path(baseline)
    return prev_restart


def _seed_restart_for_memory_handoff(
    io: CharmmTrajectoryFiles,
    kw: dict[str, Any],
    *,
    stage: MdStage,
) -> Path:
    """Persist in-memory state to ``restart_write`` for on-disk checkpoint.

    Used before overlap-chunked NPT stages after post-stage minimization (e.g.
    bonded-MM mini).  Does **not** enable ``READYN``: static ``write restart``
    lacks CPT barostat internals — callers must invoke
    :func:`_configure_npt_dynamics_start` (or heat/NVE equivalents) for in-memory
    ``dyna`` continuation.
    """
    if io.restart_write is None:
        raise RuntimeError(
            f"memory handoff for stage {stage!r} requires restart_write on I/O"
        )
    rewrite_dynamics_restart_from_current_state(io.restart_write)
    seed = Path(io.restart_write)
    kw.setdefault("new", False)
    if stage in ("equi", "prod"):
        kw.pop("firstt", None)
    return seed


def _configure_npt_dynamics_start(
    kw: dict[str, Any],
    io: CharmmTrajectoryFiles,
    *,
    coords_in_memory: bool,
    restart_from_file: bool,
    timestep_ps: float,
    use_pbc: bool,
    quiet: bool,
    temp: float,
    box_side: float | None = None,
) -> None:
    """Fresh Boltzmann draw and CPT barostat for in-memory handoff after mini/rescue.

    ``write restart`` after SD/minimization saves coordinates but not barostat
    piston state; ``READYN`` then yields garbage ``PIXX``/``PRESSE``.  Match HEAT/NVE:
    continue from RAM with ``restart=False``, ``start=False``, ``iasvel=1``.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        assign_velocities_at_temperature,
    )

    if not coords_in_memory:
        return

    io.restart_read = None
    kw["restart"] = False
    kw["new"] = False
    kw.pop("iunrea", None)
    kw["iunrea"] = -1
    target = float(kw.get("hoover reft", kw.get("treference", temp)))
    use_cpt = bool(kw.get("cpt"))
    if use_pbc and use_cpt and box_side is not None:
        ensure_charmm_crystal_for_cpt(float(box_side), quiet=quiet)
    assign_velocities_at_temperature(
        target,
        timestep_ps=timestep_ps,
        restart_path=None,
        use_pbc=use_pbc and use_cpt,
    )
    kw["start"] = False
    kw["iasvel"] = 1
    kw.pop("firstt", None)
    if not quiet:
        label = "NPT" if use_cpt else "NVT"
        print(
            f"{label}: Boltzmann velocities at {target:.1f} K "
            "(in-memory coords after mini; fresh barostat, no READYN)",
            flush=True,
        )


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
    *,
    handoff_in=None,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    from mmml.cli.run.md_handoff import (
        cluster_geometry_from_handoff,
        ensure_psf_for_handoff_cluster,
        get_handoff_in,
    )
    from mmml.cli.run.md_pbc_suite.ase import _parse_composition
    from mmml.interfaces.pycharmmInterface.mlpot.setup import load_cluster_from_artifacts

    ho = handoff_in if handoff_in is not None else get_handoff_in()
    if ho is not None:
        n_mol_default = int(getattr(args, "n_molecules", 1) or 1)
        z, r0, atoms_per_list, residue_labels, _ = cluster_geometry_from_handoff(
            ho,
            composition=getattr(args, "composition", None),
            n_molecules=n_mol_default,
        )
        if getattr(args, "composition", None):
            composition = _parse_composition(args.composition)
        else:
            n_mol = len(atoms_per_list)
            composition = [(residue_labels[0], n_mol)]
        ensure_psf_for_handoff_cluster(
            composition=composition,
            atomic_numbers=z,
            atoms_per_list=atoms_per_list,
            residue_labels=residue_labels,
            positions=r0,
            quiet=bool(getattr(args, "quiet", False)),
        )
        n_mol = len(atoms_per_list)
        tag = composition_tag(composition, getattr(args, "residue", "MEOH").upper(), n_mol)
        if not getattr(args, "quiet", False):
            print(
                f"Continuing from handoff ({len(z)} atoms); skipped Packmol/cluster build",
                flush=True,
            )
        return z, np.asarray(r0, dtype=np.float64), n_mol, tag
    if getattr(args, "skip_cluster_build", False) or getattr(args, "from_psf", None):
        return load_cluster_from_artifacts(args)
    return build_cluster_from_args_with_tag(args)


def run_staged_workflow(args: argparse.Namespace) -> int:
    from mmml.cli.run.md_handoff import get_handoff_in, handoff_from_charmm, set_handoff_out
    from mmml.cli.run.md_stage_summary import cubic_box_side_from_cell

    handoff_in = get_handoff_in()
    stages = resolve_md_stages(args)
    if handoff_in is not None and not getattr(args, "handoff_pre_minimize", False):
        stages = [s for s in stages if s != "mini"]
    if getattr(args, "no_pre_minimize", False):
        stages = [s for s in stages if s != "mini"]
    if not stages:
        raise ValueError("no MD stages selected")

    fix_resids = resolve_fix_resids(args)
    dynamics_constrain = resolve_constrain_resids(args)
    ckpt = resolve_checkpoint(args.checkpoint)
    z, r, n_mol, tag = _load_or_build_cluster(args, handoff_in=handoff_in)
    if handoff_in is not None:
        r = np.asarray(handoff_in.positions, dtype=np.float64)
        if (
            handoff_in.atomic_numbers is not None
            and len(handoff_in.atomic_numbers) == len(r)
            and int(np.asarray(handoff_in.atomic_numbers).sum()) > 0
        ):
            z = np.asarray(handoff_in.atomic_numbers, dtype=np.int32)
        if handoff_in.cell is not None:
            side = cubic_box_side_from_cell(handoff_in.cell)
            if side is not None:
                args.box_size = float(side)
    validate_resids_for_cluster(fix_resids, n_mol)
    validate_resids_for_cluster(dynamics_constrain, n_mol)
    print_cluster_geometry_summary(r, n_mol)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_atoms = len(z)
    paths = _artifact_paths(out_dir, tag)
    save_artifacts = bool(getattr(args, "save", True))
    n_heat_segments_early = max(1, int(getattr(args, "n_heat_segments", 1)))
    if not getattr(args, "restart_from", None):
        summary_path = out_dir / "stage_summary.json"
        should_resume = False
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                should_resume = int(payload.get("exit_code", 0)) != 0
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                should_resume = False
        if should_resume:
            resume_restart = discover_resume_restart(
                out_dir,
                tag,
                paths=paths,
                n_heat_segments=n_heat_segments_early,
            )
            if resume_restart is not None:
                args.restart_from = str(resume_restart)
                if not args.quiet:
                    print(
                        f"Resuming staged workflow from {resume_restart.name}",
                        flush=True,
                    )
    mini_registry: MinimizeArtifactRegistry | None = (
        MinimizeArtifactRegistry(out_dir, tag) if save_artifacts else None
    )
    legacy_mlpot = legacy_mlpot_mini_paths(out_dir, tag) if save_artifacts else None

    charmm_pbc = resolve_charmm_use_pbc(args)
    mlpot_pbc = resolve_mlpot_use_pbc(args)
    loose_pbc = resolve_loose_pbc(charmm_pbc, mlpot_pbc)
    box_side = resolve_pbc_box_side(args, r) if charmm_pbc else None
    if charmm_pbc and not args.quiet:
        if charmm_pbc and not mlpot_pbc:
            print(
                f"CHARMM loose PBC: cubic L={box_side:.3f} Å "
                "(ML open boundary; no MIC)",
                flush=True,
            )
        else:
            print(f"PBC cubic box: {box_side:.3f} Å", flush=True)

    dt_fs = float(getattr(args, "dt_fs", 0.25))
    timestep_ps = timestep_ps_from_dt_fs(dt_fs)
    if getattr(args, "timestep_ps", None) is not None:
        timestep_ps = float(args.timestep_ps)
    temp = float(getattr(args, "temperature", getattr(args, "temp", 300.0)))
    mini_nprint = apply_charmm_output_from_args(args)
    show_energy = resolve_show_energy(args)
    echeck = resolve_echeck_for_cluster(args, n_atoms=n_atoms, n_monomers=n_mol)
    mini_nstep = resolve_mini_nstep(
        args, n_mol, n_atoms=n_atoms, pbc=(mlpot_pbc or charmm_pbc)
    )
    overlap_cfg = resolve_dynamics_overlap_config(
        args,
        n_monomers=n_mol,
        use_pbc=charmm_pbc,
        fallback_box_side_A=box_side if charmm_pbc else None,
    )
    if bool(getattr(args, "save_forces_npz", False)):
        from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
            ForceCheckpointConfig,
            configure_force_checkpoint,
        )

        atoms_per = n_atoms // int(n_mol) if int(n_mol) > 0 else None
        configure_force_checkpoint(
            ForceCheckpointConfig(
                enabled=True,
                interval=int(getattr(args, "forces_npz_interval", 1) or 1),
                output_path=out_dir / "forces.npz",
                n_monomers=int(n_mol),
                atoms_per_monomer=atoms_per,
            )
        )
        if not args.quiet:
            print(
                f"Force checkpoint: {out_dir / 'forces.npz'} "
                f"every {int(getattr(args, 'forces_npz_interval', 1))} step(s)",
                flush=True,
            )
    if overlap_cfg.enabled and not args.quiet:
        print(
            f"Dynamics overlap guard: action={overlap_cfg.action}, "
            f"min_distance={overlap_cfg.min_distance_A:.2f} Å, "
            f"check every {overlap_cfg.check_interval} steps"
            + (
                ", last-resort monomer repack on"
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

    setup_charmm_environment(use_pbc=charmm_pbc, cubic_box_side_A=box_side)
    sync_charmm_positions(r)
    if handoff_in is not None:
        sync_charmm_positions(handoff_in.positions)
        from mmml.cli.run.md_handoff import prepare_pycharmm_handoff_continuation

        seed_restart = prepare_pycharmm_handoff_continuation(
            handoff_in,
            args,
            out_dir,
            paths,
            quiet=bool(args.quiet),
        )
        if seed_restart is not None:
            args.restart_from = seed_restart

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
        if mini_registry is not None:
            mini_registry.record(
                PACKMOL_CLUSTER,
                {"pdb": vmd_files["pdb"], "psf": vmd_files["psf"]},
            )
    recovery_topology_psf = vmd_topo_psf if Path(vmd_topo_psf).is_file() else None

    pretreat_mm = bool(getattr(args, "charmm_mm_pretreat", False))
    if pretreat_mm:
        r = run_charmm_mm_pretreat_before_mlpot(
            args,
            paths=paths,
            timestep_ps=timestep_ps,
            use_pbc=charmm_pbc,
            temp=temp,
            echeck=echeck,
            mini_nprint=mini_nprint,
            reference_positions=r,
            skip_minimize=handoff_in is not None,
        )
        sync_charmm_positions(r)
    elif "mini" in stages and not getattr(args, "skip_cluster_build", False):
        save_mini = bool(getattr(args, "save", True))
        mini_dcd_nsavc = resolve_dcd_nsavc_for_args(args, nstep=mini_nstep)
        r = _charmm_pre_minimize_before_mlpot(
            args,
            nprint=mini_nprint,
            reference_positions=r,
            dcd_path=paths["mini_charmm_dcd"] if save_mini else None,
            dcd_nsavc=mini_dcd_nsavc if save_mini else 0,
            save_crd_path=paths["charmm_mm_crd"] if save_mini else None,
            save_pdb_path=paths["charmm_mm_pdb"] if save_mini else None,
            save_psf_path=paths["charmm_mm_psf"] if save_mini else None,
            save_energy_json_path=paths["charmm_mm_energy_json"] if save_mini else None,
            save_title=CHARMM_MM_PRE.label,
            use_pbc=charmm_pbc,
        )
        sync_charmm_positions(r)
        if save_mini and mini_registry is not None:
            mini_registry.record(
                CHARMM_MM_PRE,
                {
                    "pdb": paths["charmm_mm_pdb"],
                    "crd": paths["charmm_mm_crd"],
                    "psf": paths["charmm_mm_psf"],
                    "energy_json": paths["charmm_mm_energy_json"],
                },
                grms_kcalmol_A=charmm_grms(),
            )

    if not mlpot_pbc:
        # Install MMFP once after Packmol / CHARMM pretreat / pre-MLpot mini so
        # droff tuning and coor orient are not repeated (stacked walls / COM shifts).
        apply_flat_bottom_from_args(args)
        r = get_charmm_positions_array()

    baseline = None
    if (
        getattr(args, "bonded_mm_mini", True)
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
        cubic_box_side_A=box_side,
        mlpot_use_pbc=mlpot_pbc,
        verbose=not args.quiet,
        args=args,
        topology_psf=recovery_topology_psf,
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
            mini_dcd_nsavc = resolve_dcd_nsavc_for_args(args, nstep=mini_nstep)
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
                    mlpot_ctx=ctx,
                    save=save_mini,
                    pdb_path=paths["mlpot_mmml_pdb"] if save_mini else None,
                    crd_path=paths["mlpot_mmml_crd"] if save_mini else None,
                    psf_path=paths["mlpot_mmml_psf"] if save_mini else None,
                    energy_json_path=paths["mlpot_mmml_energy_json"] if save_mini else None,
                    xyz_path=paths["mlpot_mmml_xyz"] if save_mini else None,
                    dcd_path=paths["mlpot_mmml_dcd"] if save_mini else None,
                    dcd_nsavc=mini_dcd_nsavc if save_mini else 0,
                    title=MLPOT_MMML.label,
                    skip_if_crd_exists=bool(getattr(args, "skip_if_crd_exists", False)),
                    test_first=resolve_test_first_config(args),
                    show_energy=show_energy,
                )
            )
            if save_mini and legacy_mlpot is not None:
                mirror_legacy_mlpot_files(
                    {
                        "pdb": paths["mlpot_mmml_pdb"],
                        "crd": paths["mlpot_mmml_crd"],
                        "psf": paths["mlpot_mmml_psf"],
                        "xyz": paths["mlpot_mmml_xyz"],
                        "energy_json": paths["mlpot_mmml_energy_json"],
                    },
                    legacy_mlpot,
                )
            if save_mini and mini_registry is not None:
                mini_registry.record(
                    MLPOT_MMML,
                    {
                        "pdb": paths["mlpot_mmml_pdb"],
                        "crd": paths["mlpot_mmml_crd"],
                        "psf": paths["mlpot_mmml_psf"],
                        "xyz": paths["mlpot_mmml_xyz"],
                        "energy_json": paths["mlpot_mmml_energy_json"],
                    },
                    grms_kcalmol_A=charmm_grms(),
                )
            sync_charmm_positions(get_charmm_positions_array())
            refresh_mlpot_energy_and_grms(
                ctx,
                context="Post MLpot mini" if not args.quiet else "Post MLpot mini GRMS",
            )
            mini_trajectories = _trajectory_outputs(paths["mini_charmm_dcd"])
            mini_trajectories.extend(_trajectory_outputs(paths["mlpot_mmml_dcd"]))
            last_traj = mini_trajectories[-1] if mini_trajectories else None
            maybe_run_bonded_mm_mini_after_stage(
                ctx,
                args,
                stage="mini",
                baseline=baseline,
                restart_path=paths["mini_crd"],
                topology_psf=recovery_topology_psf,
                mini_registry=mini_registry,
                snapshot_spec=BONDED_MM_AFTER_MINI,
                snapshot_paths=(
                    {
                        "pdb": paths["bonded_mm_after_mini_pdb"],
                        "crd": paths["bonded_mm_after_mini_crd"],
                    }
                    if save_mini
                    else None
                ),
            )

        dyn_stages = [s for s in _STAGE_ORDER if s in stages and s != "mini"]
        if not dyn_stages:
            return 0

        stage_overlap_pre = _overlap_for_stage(
            "heat",
            overlap_cfg,
            ctx=ctx,
            args=args,
            topology_psf=recovery_topology_psf,
            mini_registry=mini_registry,
        )
        overlap_rescued = False
        if stage_overlap_pre is not None and (
            stage_overlap_pre.enabled
            or stage_overlap_pre.intra_enabled
            or stage_overlap_pre.extent_enabled
        ):
            _, overlap_rescued = check_dynamics_overlap(
                stage_overlap_pre,
                context="after MLpot mini (pre-dynamics)",
                mlpot_ctx=ctx,
            )
            if overlap_rescued:
                ctx.reregister_mlpot()
                refresh_mlpot_energy_and_grms(
                    ctx,
                    context="Post overlap rescue (pre-dynamics)",
                )

        baseline_path = write_geometry_baseline_restart(out_dir, tag)
        if baseline_path is not None:
            paths["geometry_baseline_res"] = baseline_path
            if not args.quiet:
                print(
                    f"Geometry baseline restart -> {baseline_path.name}",
                    flush=True,
                )

        overlap_cfg = attach_geometry_checkpoints_to_overlap(
            overlap_cfg,
            paths=paths,
            tag=tag,
            n_heat_segments=n_heat_segments_early,
        )

        assert_mlpot_user_active(ctx, context="staged dynamics", quiet=bool(args.quiet))
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
            context="staged dynamics",
            quiet=bool(args.quiet),
        )

        if dynamics_constrain:
            setup_cons_fix_for_resids(dynamics_constrain)

        n_heat_segments = max(1, int(getattr(args, "n_heat_segments", 1)))
        n_equi_segments = max(1, int(getattr(args, "n_equi_segments", 1)))
        n_prod_segments = max(1, int(getattr(args, "n_prod_segments", 1)))
        if "equi" in dyn_stages and n_equi_segments > 1:
            equi_idx = dyn_stages.index("equi")
            dyn_stages = dyn_stages[:equi_idx] + ["equi"] * n_equi_segments
        if "prod" in dyn_stages and n_prod_segments > 1:
            prod_idx = dyn_stages.index("prod")
            dyn_stages = dyn_stages[:prod_idx] + ["prod"] * n_prod_segments

        equi_restart_for_prod = _equi_restart_name(tag, n_equi_segments)
        # After MLpot mini, coordinates live in CHARMM.  Campaign resume may set
        # ``restart_from`` to pretreat MM or a stale heat scratch before mini runs;
        # that seed must not force READYN on pre-MLpot checkpoints for MLpot heat.
        if "mini" in stages:
            restart_from = None
        prev_restart: Path | None = restart_from
        prev_restart_is_current_state = "mini" in stages
        memory_handoff_next = False
        for stage in dyn_stages:
            if stage == "heat" and n_heat_segments > 1:
                heat_firstt, heat_finalt = resolve_heat_firstt_finalt(
                    args, default_temp=temp
                )
                heat_thermostat = resolve_heat_thermostat(args)
                if not args.quiet:
                    neh = bool(getattr(args, "no_echeck_heat", False))
                    print(
                        f"HEAT policy: thermostat={heat_thermostat} "
                        f"no_echeck_heat={neh} "
                        f"({n_heat_segments} segment(s), "
                        f"{heat_firstt:.1f}→{heat_finalt:.1f} K)",
                        flush=True,
                    )
                if prev_restart_is_current_state:
                    initial = None
                else:
                    initial = _prior_restart_for_stage(
                        "heat",
                        paths,
                        restart_from=prev_restart,
                        tag=tag,
                        n_heat_segments=n_heat_segments,
                    )
                seg_chain = npt_restart_chain(
                    out_dir,
                    n_segments=n_heat_segments,
                    prefix=f"heat_{tag}",
                    initial_restart=initial,
                )
                for seg_i, seg_io in enumerate(seg_chain):
                    if (
                        seg_i > 0
                        and prev_restart_is_current_state
                        and prev_restart is not None
                    ):
                        ensure_segment_restart_checkpoint(prev_restart)
                    seg_ps = _stage_ps(args, "heat") / n_heat_segments
                    nstep = dynamics_nstep_from_ps(seg_ps, dt_fs)
                    dcd_nsavc = resolve_dcd_nsavc_for_args(
                        args, timestep_ps=timestep_ps, nstep=nstep
                    )
                    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
                    save_interval_ps = timestep_ps * dcd_nsavc
                    seg_prep_quiet = bool(args.quiet) or seg_i > 0
                    use_memory = (memory_handoff_next and seg_i == 0) or (
                        seg_i == 0 and prev_restart_is_current_state
                    )
                    if use_memory:
                        restart = False
                        rread = None
                    else:
                        rread = seg_io.restart_read
                        restart = rread is not None and Path(rread).is_file()
                        if (
                            seg_i == 0
                            and _can_seed_stage_from_memory(
                                Path(rread) if rread is not None else None,
                                prev_restart=prev_restart,
                                prev_restart_is_current_state=prev_restart_is_current_state,
                            )
                        ):
                            use_memory = True
                            restart = False
                            rread = None
                        elif seg_i > 0 and prev_restart_is_current_state:
                            # Continue in-process: avoid READYN stale CPT after overlap rescue.
                            use_memory = True
                            restart = False
                            rread = None
                    if not seg_prep_quiet:
                        print(
                            f"\nHEAT segment {seg_i + 1}/{n_heat_segments}: "
                            f"{nstep} steps @ {timestep_ps} ps "
                            f"({heat_firstt:.1f}→{heat_finalt:.1f} K ramp, "
                            f"segment bath "
                            f"{heat_firstt + (heat_finalt - heat_firstt) * (seg_i / n_heat_segments):.1f}"
                            f"→"
                            f"{heat_firstt + (heat_finalt - heat_firstt) * ((seg_i + 1) / n_heat_segments):.1f} K)"
                            + (" | memory handoff" if use_memory else ""),
                            flush=True,
                        )
                    restart_path = Path(rread) if restart and rread else None
                    kw = _build_stage_dynamics_kw(
                        "heat",
                        args=args,
                        timestep_ps=timestep_ps,
                        nstep=nstep,
                        save_interval_ps=save_interval_ps,
                        temp=temp,
                        echeck=echeck,
                        dyn_print=dyn_print,
                        restart=restart,
                        use_pbc=charmm_pbc,
                        memory_handoff=use_memory,
                    )
                    kw["nsavc"] = dcd_nsavc
                    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                        apply_dyn_inbfrq_from_args,
                        apply_heat_segment_ramp_kwargs,
                        finalize_heat_dynamics_frequencies,
                        sync_charmm_lists_after_mini,
                    )

                    apply_heat_segment_ramp_kwargs(
                        kw,
                        seg_index=seg_i,
                        n_segments=n_heat_segments,
                        heat_firstt=heat_firstt,
                        heat_finalt=heat_finalt,
                        nstep=nstep,
                        ihtfrq=resolve_heat_ihtfrq(args, nstep=nstep),
                    )
                    overlap_prior_restart = _overlap_extent_prior_restart(paths, prev_restart)
                    _sync_mlpot_cell_before_npt(
                        "heat",
                        mlpot_pbc=mlpot_pbc,
                        pyCModel=pyCModel,
                        quiet=bool(args.quiet),
                        restart_path=restart_path,
                    )
                    ensure_domdec_off_for_mlpot_energy(context="staged NPT segment")
                    if seg_i == 0 and not _heat_in_place_restart(seg_io):
                        _reset_stage_restart(
                            Path(seg_io.restart_write) if seg_io.restart_write else None,
                            trajectory_path=(
                                Path(seg_io.trajectory) if seg_io.trajectory else None
                            ),
                            restart_read=(
                                Path(seg_io.restart_read)
                                if use_memory and seg_io.restart_read is not None
                                else (
                                    Path(rread)
                                    if restart and rread is not None
                                    else None
                                )
                            ),
                        )
                        _reset_stage_trajectory(
                            Path(seg_io.trajectory) if seg_io.trajectory else None,
                            rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
                        )
                    if overlap_prior_restart is None and _should_seed_heat_prior_restart(
                        seg_i=seg_i,
                        prev_restart_is_current_state=prev_restart_is_current_state,
                        use_memory=use_memory,
                        memory_handoff_next=memory_handoff_next,
                    ):
                        overlap_prior_restart = _seed_restart_for_memory_handoff(
                            seg_io, kw, stage="heat"
                        )
                        restart_path = overlap_prior_restart
                    assert_mlpot_user_active(
                        ctx,
                        context=f"heat segment {seg_i + 1}/{n_heat_segments}",
                        quiet=seg_prep_quiet,
                    )
                    apply_comp_velocity_policy("heat", kw, args, quiet=seg_prep_quiet)
                    apply_dyn_inbfrq_from_args(kw, args, charmm_pbc=charmm_pbc)
                    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                        apply_dyn_imgfrq_from_args,
                    )

                    apply_dyn_imgfrq_from_args(kw, args, charmm_pbc=charmm_pbc)
                    if (seg_i == 0 or use_memory) and (
                        use_memory or prev_restart_is_current_state
                    ):
                        sync_charmm_lists_after_mini(quiet=seg_prep_quiet)
                    if use_memory:
                        if (
                            charmm_pbc
                            and box_side is not None
                            and heat_thermostat == "hoover"
                            and kw.get("cpt")
                        ):
                            ensure_charmm_crystal_for_cpt(
                                float(box_side),
                                quiet=seg_prep_quiet,
                            )
                        _configure_heat_dynamics_start(
                            kw,
                            seg_io,
                            coords_in_memory=True,
                            restart_from_file=False,
                            timestep_ps=timestep_ps,
                            use_pbc=charmm_pbc,
                            quiet=seg_prep_quiet,
                            heat_thermostat=heat_thermostat,
                        )
                    elif seg_i == 0:
                        if (
                            charmm_pbc
                            and box_side is not None
                            and heat_thermostat == "hoover"
                            and kw.get("cpt")
                        ):
                            ensure_charmm_crystal_for_cpt(
                                float(box_side),
                                quiet=seg_prep_quiet,
                            )
                        _configure_heat_dynamics_start(
                            kw,
                            seg_io,
                            coords_in_memory=prev_restart_is_current_state,
                            restart_from_file=restart
                            and seg_io.restart_read is not None,
                            timestep_ps=timestep_ps,
                            use_pbc=charmm_pbc,
                            quiet=seg_prep_quiet,
                            heat_thermostat=heat_thermostat,
                        )
                    elif restart:
                        kw["iasvel"] = 0
                        kw["iasors"] = 0
                        kw["start"] = False
                        kw["restart"] = True
                    heat_mode = (
                        "Hoover CPT"
                        if heat_thermostat == "hoover" and kw.get("cpt")
                        else (
                            "scale ihtfrq"
                            if int(kw.get("ihtfrq", 0) or 0) > 0
                            else f"heat/{heat_thermostat}"
                        )
                    )
                    print(
                        f"HEAT segment {seg_i + 1}/{n_heat_segments}: "
                        f"mode={heat_mode} echeck={kw.get('echeck')} "
                        f"firstt={kw.get('firstt')} finalt={kw.get('finalt')} "
                        + ("memory handoff" if use_memory else "READYN restart"),
                        flush=True,
                    )
                    if int(kw.get("ihtfrq", 0) or 0) > 0:
                        freq_changes = finalize_heat_dynamics_frequencies(kw)
                        if freq_changes and not seg_prep_quiet:
                            parts = ", ".join(
                                f"{key} {old}->{new}"
                                for key, (old, new) in sorted(freq_changes.items())
                            )
                            print(
                                f"HEAT segment {seg_i + 1}: harmonized frequencies "
                                f"({parts}); TEMINC={float(kw.get('TEMINC', 0)):.6g} K",
                                flush=True,
                            )
                    stage_overlap = overlap_config_for_stage(
                        _overlap_for_stage(
                            "heat",
                            overlap_cfg,
                            ctx=ctx,
                            args=args,
                            topology_psf=recovery_topology_psf,
                            mini_registry=mini_registry,
                        ),
                        stage="heat",
                        nstep=nstep,
                        n_segments=n_heat_segments,
                    )
                    stage_overlap = attach_prior_segment_restart(
                        stage_overlap,
                        segment_index=seg_i,
                        prev_restart=overlap_prior_restart,
                        out_dir=out_dir,
                        restart_prefix=f"heat_{tag}",
                        restart_write=seg_io.restart_write,
                    )
                    run_dynamics_with_io(
                        kw,
                        seg_io,
                        overlap=stage_overlap,
                        overlap_context=(
                            f"heat segment {seg_i + 1}/{n_heat_segments}"
                        ),
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                        loose_pbc=loose_pbc,
                        **overlap_run_state_kwargs_from_args(args),
                    )
                    _validate_dyn_stage_completion(
                        args,
                        stage="heat",
                        nstep=nstep,
                        nsavc=dcd_nsavc,
                        io=seg_io,
                        segment_note=(
                            f"segment {seg_i + 1}/{n_heat_segments}"
                            if n_heat_segments > 1
                            else None
                        ),
                    )
                    ensure_segment_restart_checkpoint(seg_io.restart_write)
                    if baseline is not None or seg_i == n_heat_segments - 1:
                        memory_handoff_next = maybe_run_bonded_mm_mini_after_stage(
                            ctx,
                            args,
                            stage="heat",
                            baseline=baseline,
                            restart_path=seg_io.restart_write,
                            topology_psf=recovery_topology_psf,
                            mini_registry=mini_registry,
                            snapshot_spec=(
                                BONDED_MM_AFTER_HEAT
                                if seg_i == n_heat_segments - 1
                                else None
                            ),
                            snapshot_paths=(
                                {
                                    "pdb": paths["bonded_mm_after_heat_pdb"],
                                    "crd": paths["bonded_mm_after_heat_crd"],
                                }
                                if save_artifacts and seg_i == n_heat_segments - 1
                                else None
                            ),
                        )
                    else:
                        memory_handoff_next = False
                    prev_restart = seg_io.restart_write
                    prev_restart_is_current_state = True
                    last_restart_path = prev_restart
                    last_traj = seg_io.trajectory
                continue

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
                    if (
                        seg_i > 0
                        and prev_restart_is_current_state
                        and prev_restart is not None
                    ):
                        ensure_segment_restart_checkpoint(prev_restart)
                    seg_ps = _stage_ps(args, "equi") / n_equi_segments
                    nstep = dynamics_nstep_from_ps(seg_ps, dt_fs)
                    dcd_nsavc = resolve_dcd_nsavc_for_args(
                        args, timestep_ps=timestep_ps, nstep=nstep
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
                        use_pbc=charmm_pbc,
                        npt_include_firstt=(seg_i == 0),
                        memory_handoff=use_memory,
                    )
                    kw["nsavc"] = dcd_nsavc
                    if use_memory:
                        restart_path = _seed_restart_for_memory_handoff(seg_io, kw, stage="equi")
                        _configure_npt_dynamics_start(
                            kw,
                            seg_io,
                            coords_in_memory=True,
                            restart_from_file=False,
                            timestep_ps=timestep_ps,
                            use_pbc=charmm_pbc,
                            quiet=bool(args.quiet),
                            temp=temp,
                            box_side=box_side,
                        )
                    _sync_mlpot_cell_before_npt(
                        "equi",
                        mlpot_pbc=mlpot_pbc,
                        pyCModel=pyCModel,
                        quiet=bool(args.quiet),
                        restart_path=restart_path,
                    )
                    ensure_domdec_off_for_mlpot_energy(context="staged NPT segment")
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
                        overlap=_overlap_for_stage(
                            "equi",
                            overlap_cfg,
                            ctx=ctx,
                            args=args,
                            topology_psf=recovery_topology_psf,
                            mini_registry=mini_registry,
                        ),
                        overlap_context=f"equi segment {seg_i + 1}/{n_equi_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                        loose_pbc=loose_pbc,
                        **overlap_run_state_kwargs_from_args(args),
                    )
                    _validate_dyn_stage_completion(
                        args,
                        stage="equi",
                        nstep=nstep,
                        nsavc=dcd_nsavc,
                        io=seg_io,
                    )
                    ensure_segment_restart_checkpoint(seg_io.restart_write)
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
                    if (
                        seg_i > 0
                        and prev_restart_is_current_state
                        and prev_restart is not None
                    ):
                        ensure_segment_restart_checkpoint(prev_restart)
                    seg_ps = _stage_ps(args, "prod") / n_prod_segments
                    nstep = dynamics_nstep_from_ps(seg_ps, dt_fs)
                    dcd_nsavc = resolve_dcd_nsavc_for_args(
                        args, timestep_ps=timestep_ps, nstep=nstep
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
                        use_pbc=charmm_pbc,
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
                        _configure_npt_dynamics_start(
                            kw,
                            seg_io,
                            coords_in_memory=True,
                            restart_from_file=False,
                            timestep_ps=timestep_ps,
                            use_pbc=charmm_pbc,
                            quiet=bool(args.quiet),
                            temp=temp,
                            box_side=box_side,
                        )
                    _sync_mlpot_cell_before_npt(
                        "prod",
                        mlpot_pbc=mlpot_pbc,
                        pyCModel=pyCModel,
                        quiet=bool(args.quiet),
                        restart_path=restart_path,
                    )
                    ensure_domdec_off_for_mlpot_energy(context="staged NPT segment")
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
                        overlap=_overlap_for_stage(
                            "prod",
                            overlap_cfg,
                            ctx=ctx,
                            args=args,
                            topology_psf=recovery_topology_psf,
                            mini_registry=mini_registry,
                        ),
                        overlap_context=f"prod segment {seg_i + 1}/{n_prod_segments}",
                        mlpot_ctx=ctx,
                        rng_base=getattr(args, "seed", None),
                        loose_pbc=loose_pbc,
                        **overlap_run_state_kwargs_from_args(args),
                    )
                    _validate_dyn_stage_completion(
                        args,
                        stage="prod",
                        nstep=nstep,
                        nsavc=dcd_nsavc,
                        io=seg_io,
                    )
                    ensure_segment_restart_checkpoint(seg_io.restart_write)
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
            dcd_nsavc = resolve_dcd_nsavc_for_args(
                args, timestep_ps=timestep_ps, nstep=nstep
            )
            dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
            save_interval_ps = timestep_ps * dcd_nsavc

            use_memory = memory_handoff_next
            if use_memory:
                restart = False
                rread = None
            else:
                rread = prev_restart or _prior_restart_for_stage(
                    stage,
                    paths,
                    restart_from=None,
                    tag=tag,
                    n_heat_segments=n_heat_segments,
                )
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
            overlap_prior_restart = _overlap_extent_prior_restart(paths, prev_restart)

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
                use_pbc=charmm_pbc,
                memory_handoff=use_memory,
            )
            kw["nsavc"] = dcd_nsavc
            if stage == "heat" and overlap_prior_restart is None and _should_seed_heat_prior_restart(
                seg_i=0,
                prev_restart_is_current_state=prev_restart_is_current_state,
                use_memory=use_memory,
                memory_handoff_next=False,
            ):
                overlap_prior_restart = _seed_restart_for_memory_handoff(
                    io, kw, stage="heat"
                )
                restart_path = overlap_prior_restart
            elif use_memory:
                restart_path = _seed_restart_for_memory_handoff(io, kw, stage=stage)
                if stage in ("equi", "prod"):
                    _configure_npt_dynamics_start(
                        kw,
                        io,
                        coords_in_memory=True,
                        restart_from_file=False,
                        timestep_ps=timestep_ps,
                        use_pbc=charmm_pbc,
                        quiet=bool(args.quiet),
                        temp=temp,
                        box_side=box_side,
                    )
            _sync_mlpot_cell_before_npt(
                stage,
                mlpot_pbc=mlpot_pbc,
                pyCModel=pyCModel,
                quiet=bool(args.quiet),
                restart_path=restart_path,
            )
            ensure_domdec_off_for_mlpot_energy(context="staged dynamics")
            _reset_stage_restart(
                Path(io.restart_write) if io.restart_write else None,
                trajectory_path=Path(io.trajectory) if io.trajectory else None,
                restart_read=(
                    Path(io.restart_write)
                    if use_memory and io.restart_write is not None
                    else (Path(rread) if restart and rread is not None else None)
                ),
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
            from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                apply_dyn_imgfrq_from_args,
                apply_dyn_inbfrq_from_args,
            )

            apply_dyn_inbfrq_from_args(kw, args, charmm_pbc=charmm_pbc)
            apply_dyn_imgfrq_from_args(kw, args, charmm_pbc=charmm_pbc)
            if stage in ("heat", "nve") and (
                stage == "heat"
                or (
                    not charmm_pbc
                    and getattr(args, "pre_nve_charmm_update", True)
                )
            ):
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                    sync_charmm_lists_after_mini,
                )

                sync_charmm_lists_after_mini(quiet=bool(args.quiet))
            if stage == "heat":
                heat_thermostat = resolve_heat_thermostat(args)
                if (
                    charmm_pbc
                    and box_side is not None
                    and heat_thermostat == "hoover"
                    and kw.get("cpt")
                ):
                    ensure_charmm_crystal_for_cpt(
                        float(box_side),
                        quiet=bool(args.quiet),
                    )
                _configure_heat_dynamics_start(
                    kw,
                    io,
                    coords_in_memory=use_memory or prev_restart_is_current_state,
                    restart_from_file=restart and io.restart_read is not None,
                    timestep_ps=timestep_ps,
                    use_pbc=charmm_pbc,
                    quiet=bool(args.quiet),
                    heat_thermostat=heat_thermostat,
                )
            elif stage == "nve":
                nve_t = resolve_nve_boltzmann_temp(args, default_temp=temp)
                _configure_nve_dynamics_start(
                    kw,
                    io,
                    coords_in_memory=use_memory or prev_restart_is_current_state,
                    restart_from_file=restart and io.restart_read is not None,
                    timestep_ps=timestep_ps,
                    use_pbc=charmm_pbc,
                    quiet=bool(args.quiet),
                    temp=nve_t,
                )
            if stage == "heat" and not args.quiet:
                if heat_thermostat == "hoover" and kw.get("cpt"):
                    print(
                        f"HEAT Hoover (CPT): {kw.get('firstt')} -> {kw.get('finalt')} K "
                        f"over {stage_ps} ps | hoover reft={kw.get('hoover reft')} K "
                        f"tmass={kw.get('tmass')} | pmass=0 | ihtfrq=0",
                        flush=True,
                    )
                elif heat_thermostat == "hoover":
                    print(
                        f"HEAT Hoover (vacuum fallback): {kw.get('firstt')} -> "
                        f"{kw.get('finalt')} K over {stage_ps} ps | CPT Hoover needs "
                        f"CRYSTal — using ihtfrq={kw.get('ihtfrq')} scale (iasors=0)",
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
            stage_overlap = overlap_config_for_stage(
                _overlap_for_stage(
                    stage,
                    overlap_cfg,
                    ctx=ctx,
                    args=args,
                    topology_psf=recovery_topology_psf,
                    mini_registry=mini_registry,
                ),
                stage=stage,
                nstep=nstep,
                n_segments=n_heat_segments if stage == "heat" else 1,
            )
            if (
                stage == "heat"
                and n_heat_segments <= 1
                and stage_overlap is not None
                and stage_overlap.heat_segment_boundary_only
                and stage_overlap.enabled
                and int(stage_overlap.check_interval) >= nstep
                and not args.quiet
            ):
                print(
                    f"overlap (HEAT): one integration segment ({nstep} steps); "
                    "geometry check after heat completes",
                    flush=True,
                )
            if stage == "heat" and int(kw.get("ihtfrq", 0) or 0) > 0:
                from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
                    finalize_heat_dynamics_frequencies,
                )

                freq_changes = finalize_heat_dynamics_frequencies(kw)
                if freq_changes and not args.quiet:
                    parts = ", ".join(
                        f"{key} {old}->{new}"
                        for key, (old, new) in sorted(freq_changes.items())
                    )
                    print(
                        f"HEAT: harmonized dynamics frequencies for nstep={nstep} "
                        f"({parts}); TEMINC={float(kw.get('TEMINC', 0)):.6g} K",
                        flush=True,
                    )
            stage_overlap = attach_prior_segment_restart(
                stage_overlap,
                prev_restart=overlap_prior_restart,
                restart_write=io.restart_write,
            )
            run_dynamics_with_io(
                kw,
                io,
                overlap=stage_overlap,
                overlap_context=stage.upper(),
                mlpot_ctx=ctx,
                rng_base=getattr(args, "seed", None),
                loose_pbc=loose_pbc,
                **overlap_run_state_kwargs_from_args(args),
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
                mini_registry=mini_registry,
                snapshot_spec=BONDED_MM_AFTER_HEAT if stage == "heat" else None,
                snapshot_paths=(
                    {
                        "pdb": paths["bonded_mm_after_heat_pdb"],
                        "crd": paths["bonded_mm_after_heat_crd"],
                    }
                    if save_artifacts and stage == "heat"
                    else None
                ),
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
    from mmml.interfaces.pycharmmInterface.mlpot.force_checkpoint import (
        flush_force_checkpoint,
    )

    forces_path = flush_force_checkpoint()
    if forces_path is not None and not args.quiet:
        print(f"Force checkpoint saved: {forces_path}", flush=True)
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
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_last_step,
    )

    set_handoff_out(
        handoff_from_charmm(
            z,
            restart_path=last_restart_path,
            fallback_box_side_A=getattr(args, "box_size", None),
            temperature_K=float(getattr(args, "temperature", getattr(args, "temp", 300.0))),
            pressure_atm=float(args.pressure) if getattr(args, "pressure", None) is not None else None,
            step=(
                read_restart_last_step(last_restart_path)
                if last_restart_path is not None
                else None
            ),
            metadata={"backend": "pycharmm", "tag": tag, "stages": list(stages)},
        )
    )
    return 0
