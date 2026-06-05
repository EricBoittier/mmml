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
    resolve_fix_resids,
    resolve_pbc_box_side,
    resolve_show_energy,
    resolve_test_first_config,
    resolve_charmm_use_pbc,
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
    disable_charmm_domdec,
    setup_default_nbonds,
    sync_charmm_positions,
)
from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment

Phase = Literal["minimize", "dynamics", "full", "staged"]
Ensemble = Literal["nve", "nvt"]


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
) -> np.ndarray:
    """CGENFF minimize + CHARMM heat on the built cluster before :func:`register_mlpot`.

    Uses full MM terms (``apply_charmm_mm_block``), not MLpot USER. Intended for
    relaxing Packmol clashes and a short classical heat before ML dynamics.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.block_terms import apply_charmm_mm_block
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )

    n_heat = max(1, int(getattr(args, "charmm_mm_pretreat_heat_nstep", 2000)))
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
        print(
            f"\nCHARMM MM pretreat (no MLpot): SD={n_sd} ABNR={n_abnr}, "
            f"heat nstep={n_heat}",
            flush=True,
        )

    apply_charmm_mm_block()
    if not use_pbc:
        setup_default_nbonds()

    save = bool(getattr(args, "save", True))
    heat_dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=n_heat)
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

    heat_firstt, heat_finalt = resolve_heat_firstt_finalt(args, default_temp=temp)
    save_interval_ps = timestep_ps * max(
        1,
        resolve_dcd_nsavc(
            dcd_nsavc=args.dcd_nsavc,
            dcd_interval_ps=getattr(args, "dcd_interval_ps", None),
            timestep_ps=timestep_ps,
            nstep=n_heat,
        ),
    )
    kw = build_heat_dynamics(
        timestep_ps=timestep_ps,
        duration_ps=n_heat * timestep_ps,
        save_interval_ps=save_interval_ps,
        temp=temp,
        firstt=heat_firstt,
        finalt=heat_finalt,
        echeck=echeck,
        use_pbc=use_pbc,
        ihtfrq=resolve_heat_ihtfrq(args, nstep=n_heat),
    )
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=n_heat)
    kw["nstep"] = n_heat
    kw["nprint"] = dyn_print["nprint"]
    kw["iprfrq"] = dyn_print["iprfrq"]
    kw["isvfrq"] = dyn_print["isvfrq"]
    kw["iasors"] = 0
    kw["iasvel"] = 1
    apply_heat_ramp_frequencies(
        kw, nstep=n_heat, ihtfrq=resolve_heat_ihtfrq(args, nstep=n_heat)
    )
    assign_velocities_at_temperature(
        float(heat_firstt),
        timestep_ps=timestep_ps,
        restart_path=None,
        use_pbc=use_pbc,
    )
    kw["restart"] = False
    kw["new"] = False
    kw["start"] = False

    io = CharmmTrajectoryFiles(
        restart_write=paths["charmm_mm_heat_res"],
        trajectory=paths["charmm_mm_heat_dcd"] if save else None,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    if save and io.trajectory is not None:
        _reset_stage_trajectory(
            Path(io.trajectory),
            rescue_old=bool(getattr(args, "rescue_old_dcd", False)),
        )
    if not args.quiet:
        print(
            f"CHARMM MM pretreat heat: {heat_firstt:.1f} -> {heat_finalt:.1f} K, "
            f"{n_heat} steps @ {timestep_ps} ps | ihtfrq={kw.get('ihtfrq')}",
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

    atoms = ase.Atoms(numbers=z, positions=r)
    apm = (
        list(atoms_per_monomer)
        if atoms_per_monomer is not None
        else _atoms_per_monomer_list(z, n_monomers)
    )
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
    )
    if int(n_monomers) > 1:
        from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
            DecomposedMlpotModel,
            warmup_decomposed_mlpot,
        )

        if isinstance(pyCModel, DecomposedMlpotModel):
            warmup_decomposed_mlpot(
                pyCModel,
                r,
                cell=ml_cell,
                verbose=verbose,
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
    ctx = register_mlpot(
        pyCModel,
        z,
        select_all_atoms(),
        use_pbc=mlpot_use_pbc,
        mm_internal_scale=mm_internal_scale,
    )
    ctx.ml_Z = np.asarray(z, dtype=int)
    ctx.use_pbc = bool(mlpot_use_pbc)
    ctx.cubic_box_side_A = float(cubic_box_side_A) if mlpot_use_pbc and cubic_box_side_A else None
    ctx.charmm_cubic_box_side_A = (
        float(cubic_box_side_A) if cubic_box_side_A is not None else None
    )
    if mlpot_use_pbc and cubic_box_side_A is not None:
        refresh_nbonds_after_mlpot_pbc(
            cubic_box_side_A=float(cubic_box_side_A),
            force=True,
        )
    sync_charmm_positions(r)
    pos_chk = get_charmm_positions_array()
    if np.allclose(pos_chk, 0.0):
        sync_charmm_positions(r)
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
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import resolve_charmm_cubic_box_side_A

    if restart_path is not None:
        rpath = Path(restart_path)
        if isinstance(pyCModel, DecomposedMlpotModel):
            pyCModel._npt_restart_read = rpath if rpath.is_file() else None

    if fallback_side_A is None:
        old_cell = getattr(pyCModel, "_cell", False)
        if old_cell:
            fallback_side_A = float(old_cell)

    side, source = resolve_charmm_cubic_box_side_A(
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
    dyn_print = resolve_dynamics_print_kwargs(args, nstep=nstep)
    echeck = resolve_echeck_for_cluster(args, n_atoms=n_atoms, n_monomers=n_mol)
    dcd_nsavc = resolve_dcd_nsavc(
        dcd_nsavc=args.dcd_nsavc,
        dcd_interval_ps=args.dcd_interval_ps,
        timestep_ps=timestep_ps,
        nstep=nstep,
    )

    stem = "nve" if ensemble == "nve" else "nvt"
    res_path = out_dir / f"{stem}_{tag}.res"
    dcd_path = out_dir / f"{stem}_{tag}.dcd"
    save = bool(getattr(args, "save", True))
    charmm_mini_dcd_path = out_dir / f"mini_charmm_mm_{tag}.dcd"
    mlpot_mini_dcd_path = out_dir / f"mini_full_mlpot_{tag}.dcd"

    if pre_minimize is None:
        pre_minimize = not getattr(args, "no_pre_minimize", False)
    mini_nstep = resolve_mini_nstep(args, n_mol)
    mini_dcd_nsavc = resolve_dcd_nsavc(dcd_nsavc=args.dcd_nsavc, nstep=mini_nstep)
    charmm_pbc = resolve_charmm_use_pbc(args)
    mlpot_pbc = resolve_mlpot_use_pbc(args)
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
    )
    show_energy = resolve_show_energy(args)

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
            grms = charmm_grms()
            if not args.quiet:
                print(f"Post MLpot mini GRMS: {grms:.4f} kcal/mol/Å")

        # MMFP flat-bottom for dynamics only (avoid fighting SD on the initial Packmol cloud).
        apply_flat_bottom_from_args(args)

        assert_mlpot_user_active(ctx, context="dynamics", quiet=bool(args.quiet))
        max_grms = float(getattr(args, "max_grms_before_dyn", 50.0))
        assert_dynamics_ready(
            max_grms=max_grms,
            abort=not getattr(args, "allow_high_grms", False),
            require_mlpot_user=True,
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
        disable_charmm_domdec()
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

            stage_overlap = overlap_config_for_stage(
                overlap_cfg,
                stage="heat",
                nstep=nstep,
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


def run_workflow(
    args: argparse.Namespace,
    *,
    phase: Phase,
    ensemble: Ensemble = "nve",
) -> int:
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
