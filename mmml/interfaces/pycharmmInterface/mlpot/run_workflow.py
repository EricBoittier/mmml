"""In-process CHARMM MLpot minimize / dynamics workflows (``mmml md-system --backend pycharmm``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

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
    apply_flat_bottom_from_args,
    assert_dynamics_ready,
    charmm_grms,
    resolve_echeck_for_cluster,
    resolve_mini_nstep,
    resolve_fix_resids,
    resolve_show_energy,
    setup_cons_fix_for_resids,
    timestep_ps_from_dt_fs,
    turn_off_cons_fix,
    validate_resids_for_cluster,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    CharmmMmMinimizeConfig,
    CharmmTrajectoryFiles,
    MinimizeWithMlpotConfig,
    build_heat_dynamics,
    build_nve_dynamics,
    minimize_charmm_mm_only,
    minimize_with_mlpot,
    run_dynamics_with_io,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    get_charmm_positions_array,
    load_physnet_mlpot_bundle,
    register_mlpot,
    save_cluster_topology_for_vmd,
    select_all_atoms,
    select_by_resids,
    disable_charmm_domdec,
    setup_default_nbonds,
    sync_charmm_positions,
)

Phase = Literal["minimize", "dynamics", "full"]
Ensemble = Literal["nve", "nvt"]


def _charmm_pre_minimize_before_mlpot(
    args: argparse.Namespace,
    *,
    nprint: int,
    reference_positions: np.ndarray | None = None,
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
        )
    )
    r_mm = get_charmm_positions_array()
    grms = charmm_grms()
    if not args.quiet:
        print(f"Post MM pre-min GRMS: {grms:.4f} kcal/mol/Å", flush=True)
    return r_mm


def _atoms_per_monomer_list(z: np.ndarray, n_monomers: int) -> list[int]:
    n_atoms = int(len(z))
    if n_atoms % int(n_monomers) != 0:
        raise ValueError(
            f"atom count {n_atoms} not divisible by n_monomers={n_monomers}"
        )
    per = n_atoms // int(n_monomers)
    return [per] * int(n_monomers)


def _register_mlpot_context(
    z: np.ndarray,
    r: np.ndarray,
    ckpt: Path,
    n_atoms: int,
    n_monomers: int,
    *,
    verbose: bool = False,
):
    import ase

    atoms = ase.Atoms(numbers=z, positions=r)
    _, _, pyCModel = load_physnet_mlpot_bundle(
        ckpt,
        n_atoms,
        atoms,
        n_monomers=n_monomers,
        atoms_per_monomer=_atoms_per_monomer_list(z, n_monomers),
        verbose=verbose,
    )
    ctx = register_mlpot(pyCModel, z, select_all_atoms())
    sync_charmm_positions(r)
    pos_chk = get_charmm_positions_array()
    if np.allclose(pos_chk, 0.0):
        sync_charmm_positions(r)
    return ctx, pyCModel


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
    save = bool(getattr(args, "save", True))

    setup_default_nbonds()
    sync_charmm_positions(r)
    vmd_topo_psf = out_dir / f"cluster_for_vmd_{tag}.psf"
    if not getattr(args, "no_save_vmd_topology", False):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]

    r = _charmm_pre_minimize_before_mlpot(
        args, nprint=nprint, reference_positions=r
    )
    sync_charmm_positions(r)

    ctx, pyCModel = _register_mlpot_context(
        z, r, ckpt, n_atoms, n_mol, verbose=not args.quiet
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
                save=save,
                pdb_path=pdb_path if save else None,
                crd_path=crd_path if save else None,
                psf_path=psf_path if save else None,
                energy_json_path=energy_json_path if save else None,
                xyz_path=xyz_path if save else None,
                dcd_path=dcd_path if save else None,
                dcd_nsavc=dcd_nsavc if save else 0,
                skip_if_crd_exists=False,
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
            trajectory=dcd_path if dcd_path.is_file() else None,
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

    if pre_minimize is None:
        pre_minimize = not getattr(args, "no_pre_minimize", False)
    mini_nstep = resolve_mini_nstep(args, n_mol)

    setup_default_nbonds()
    sync_charmm_positions(r)
    vmd_topo_psf = out_dir / f"cluster_for_vmd_{tag}.psf"
    if not getattr(args, "no_save_vmd_topology", False):
        vmd_files = save_cluster_topology_for_vmd(
            out_dir, r, stem=f"cluster_for_vmd_{tag}", title="pre-MLpot cluster"
        )
        vmd_topo_psf = vmd_files["psf"]

    r = _charmm_pre_minimize_before_mlpot(
        args, nprint=mini_nprint, reference_positions=r
    )
    sync_charmm_positions(r)

    ctx, pyCModel = _register_mlpot_context(
        z, r, ckpt, n_atoms, n_mol, verbose=not args.quiet
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
                    save=False,
                    show_energy=show_energy,
                    skip_if_crd_exists=False,
                )
            )
            sync_charmm_positions(get_charmm_positions_array())
            grms = charmm_grms()
            if not args.quiet:
                print(f"Post MLpot mini GRMS: {grms:.4f} kcal/mol/Å")

        # MMFP flat-bottom for dynamics only (avoid fighting SD on the initial Packmol cloud).
        apply_flat_bottom_from_args(args)

        max_grms = float(getattr(args, "max_grms_before_dyn", 50.0))
        assert_dynamics_ready(
            max_grms=max_grms,
            abort=not getattr(args, "allow_high_grms", False),
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
            )
            kw["nprint"] = dyn_print["nprint"]
            kw["iprfrq"] = dyn_print["iprfrq"]
            kw["isvfrq"] = dyn_print["isvfrq"]
        kw["new"] = True
        kw["start"] = True
        kw["nstep"] = nstep
        kw["nsavc"] = dcd_nsavc
        print(
            f"DCD nsavc={dcd_nsavc} ({dcd_nsavc * timestep_ps:.6f} ps/frame) | "
            f"dyn print every {dyn_print['nprint']} steps | echeck={echeck} kcal/mol"
        )
        disable_charmm_domdec()
        run_dynamics_with_io(kw, io)
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
        trajectory=dcd_path,
        n_atoms=n_atoms,
    )
    return 0


def run_workflow(
    args: argparse.Namespace,
    *,
    phase: Phase,
    ensemble: Ensemble = "nve",
) -> int:
    if phase == "minimize":
        return run_minimize_workflow(args)
    if phase == "dynamics":
        return run_dynamics_workflow(args, ensemble=ensemble)
    if phase == "full":
        return run_dynamics_workflow(args, ensemble=ensemble, pre_minimize=True)
    raise ValueError(f"Unknown phase: {phase}")
