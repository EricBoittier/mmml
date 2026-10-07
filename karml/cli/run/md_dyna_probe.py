"""One-step CHARMM DYNA probe: compare force lanes pre/post integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mmml.cli.run.md_evaluate_npz import (
    _mmml_cutoff_args,
    _prepare_evaluate_npz_context,
    classify_ml_regime,
    compare_evaluate_to_reference_npz,
    enrich_compare_with_force_sources,
    reference_metrics_at_eval_geometry,
    reference_metrics_from_npz,
    resolve_reference_units,
    setup_pycharmm_eval_mlpot,
)
from mmml.interfaces.pycharmmInterface.hybrid_reference import compute_com_distances
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol


def _jsonable_force_sources(force_sources: dict[str, np.ndarray]) -> dict[str, list]:
    return {
        name: np.asarray(forces, dtype=np.float64).reshape(-1, 3).tolist()
        for name, forces in force_sources.items()
    }


def pycharmm_dyna_snapshot(
    ctx: Any,
    calc: Any,
    *,
    z: np.ndarray,
    n_monomers: int,
    use_pbc: bool,
    L: float | None,
    label: str,
    step: int,
    quiet: bool = False,
    reference_path: Path | None = None,
    ref_frame: int | None = None,
    ref_energy_unit: str | None = None,
    ref_force_unit: str | None = None,
    reference: Any | None = None,
) -> dict[str, Any]:
    """CHARMM state after ``ENER FORCE``: energies, all force lanes, optional MP2 compare."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_energy_row,
        charmm_positions_angstrom,
        collect_evaluate_force_sources_ev_angstrom,
        cross_lane_force_rmse_ev_angstrom,
        refresh_mlpot_energy_and_grms,
        resolve_evaluate_forces_ev_angstrom,
    )

    refresh_ctx = "" if quiet else f"dyna-probe {label}"
    grms = refresh_mlpot_energy_and_grms(
        ctx,
        context=refresh_ctx,
        reregister=False,
    )
    pos = charmm_positions_angstrom()[: int(len(z))]
    charmm_row = charmm_energy_row()
    total_kcal = float(charmm_row.get("ENER", charmm_row.get("ENERGY", 0.0)))
    energy_eV = total_kcal / float(ev2kcalmol)
    force_sources = collect_evaluate_force_sources_ev_angstrom(
        calc,
        natom=int(len(z)),
        positions=pos,
        use_pbc=use_pbc,
        box_A=L,
    )
    forces_ev, force_source = resolve_evaluate_forces_ev_angstrom(
        calc,
        natom=int(len(z)),
        positions=pos,
        use_pbc=use_pbc,
        box_A=L,
    )
    snap: dict[str, Any] = {
        "label": label,
        "step": int(step),
        "energy_eV": float(energy_eV),
        "energy_kcal_mol": float(total_kcal),
        "grms_kcal_mol_A": float(grms),
        "force_source": force_source,
        "charmm_energy_terms_kcal_mol": charmm_row,
        "force_sources_ev_A": _jsonable_force_sources(force_sources),
        "cross_lane_force_rmse_eV_A": cross_lane_force_rmse_ev_angstrom(
            force_sources,
            baseline="spherical_fn",
        ),
        "n_atoms": int(len(z)),
    }
    n_atoms_monomer = int(len(z) // n_monomers)
    com_dist = float(
        compute_com_distances(
            pos.reshape(1, -1, 3),
            n_atoms_monomer=n_atoms_monomer,
            n_monomers=n_monomers,
        )[0]
    )
    snap["com_dist_A"] = com_dist

    ref_f_ev: np.ndarray | None = None
    if reference is not None and ref_frame is not None:
        ref_e_ev, ref_f_ev = reference_metrics_at_eval_geometry(
            reference,
            ref_frame=int(ref_frame),
            atomic_numbers=z,
            positions=pos,
        )
        if ref_e_ev is not None:
            snap["reference_energy_eV"] = float(ref_e_ev)
            snap["delta_energy_eV"] = float(energy_eV) - float(ref_e_ev)
    elif reference_path is not None and ref_frame is not None:
        ref_e_ev, ref_f_ev = reference_metrics_from_npz(
            reference_path,
            ref_frame=int(ref_frame),
            atomic_numbers=z,
            positions=pos,
            reference_energy_unit=ref_energy_unit,
            reference_force_unit=ref_force_unit,
        )
        if ref_e_ev is not None:
            snap["reference_energy_eV"] = float(ref_e_ev)
            snap["delta_energy_eV"] = float(energy_eV) - float(ref_e_ev)

    if reference_path is not None and ref_frame is not None:
        try:
            ref_cmp = compare_evaluate_to_reference_npz(
                reference_path,
                frame=int(ref_frame),
                atomic_numbers=z,
                positions=pos,
                energy_eV=energy_eV,
                forces_eV_A=forces_ev,
                reference_energy_unit=ref_energy_unit,
                reference_force_unit=ref_force_unit,
            )
            ref_cmp["status"] = "ok"
            ref_cmp["force_source"] = force_source
            enrich_compare_with_force_sources(
                ref_cmp,
                reference_forces_ev=ref_f_ev,
                force_sources=force_sources,
                charmm_energy_terms_kcal_mol=charmm_row,
            )
            snap["reference_compare"] = ref_cmp
        except Exception as exc:
            snap["reference_compare"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    return snap


def run_dyna_probe(args: Any) -> int:
    """Run a short NVE DYNA and record CHARMM force lanes pre/post integration."""
    if not getattr(args, "evaluate_npz", None):
        raise ValueError("--dyna-probe requires --evaluate-npz for geometry")
    backend = str(getattr(args, "backend", "auto"))
    if backend not in ("pycharmm", "auto"):
        raise ValueError("--dyna-probe requires --backend pycharmm")
    if backend == "auto":
        args.backend = "pycharmm"

    from mmml.interfaces.pycharmmInterface.charmm_mpi import prepare_serial_charmm_mpi_env
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import apply_jax_compile_xla_flags
    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
        apply_mlpot_jax_compilation_cache_env,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import build_nve_dynamics, run_dynamics
    from mmml.interfaces.pycharmmInterface.mlpot.setup import assert_mlpot_user_active

    prepare_serial_charmm_mpi_env()
    apply_mlpot_jax_compilation_cache_env(quiet=True)
    apply_jax_compile_xla_flags(quiet=True)

    ctx_prep = _prepare_evaluate_npz_context(args)
    z = ctx_prep["z"]
    n_monomers = ctx_prep["n_monomers"]
    use_pbc = ctx_prep["use_pbc"]
    L = ctx_prep["L"]
    atoms = ctx_prep["atoms"]
    reference = ctx_prep.get("reference")
    frame = int(ctx_prep["frame"])
    quiet = bool(getattr(args, "quiet", False))

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    mlpot_ctx, calc = setup_pycharmm_eval_mlpot(
        args,
        z=z,
        positions=positions,
        n_monomers=n_monomers,
        use_pbc=use_pbc,
        L=L,
    )
    assert_mlpot_user_active(mlpot_ctx, context="dyna-probe", quiet=quiet)

    ref_path = getattr(args, "evaluate_reference_npz", None) or getattr(
        args, "reference_npz", None
    )
    ref_path_resolved = (
        Path(ref_path).expanduser().resolve() if ref_path is not None else None
    )
    ref_frame = int(getattr(args, "evaluate_reference_frame", frame) or frame)
    ref_e_unit, ref_f_unit = (
        resolve_reference_units(ref_path, args) if ref_path is not None else (None, None)
    )

    ml_w, mm_on, mm_w = _mmml_cutoff_args(args)

    pre = pycharmm_dyna_snapshot(
        mlpot_ctx,
        calc,
        z=z,
        n_monomers=n_monomers,
        use_pbc=use_pbc,
        L=L,
        label="pre_dyna",
        step=0,
        quiet=quiet,
        reference_path=ref_path_resolved,
        ref_frame=ref_frame if ref_path_resolved is not None else None,
        ref_energy_unit=ref_e_unit,
        ref_force_unit=ref_f_unit,
        reference=reference,
    )
    pre["ml_regime"] = classify_ml_regime(
        float(pre["com_dist_A"]),
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
    )

    nstep = max(1, int(getattr(args, "dyna_probe_nstep", 1) or 1))
    dt_fs = float(getattr(args, "dyna_probe_dt_fs", 0.5) or 0.5)
    timestep_ps = dt_fs * 1e-3
    duration_ps = nstep * timestep_ps
    echeck = -1.0 if getattr(args, "no_echeck", False) else 10_000.0
    dyn_kw = build_nve_dynamics(
        timestep_ps=timestep_ps,
        duration_ps=duration_ps,
        save_interval_ps=timestep_ps,
        restart=False,
        use_pbc=use_pbc,
        echeck=echeck,
        nprint=max(1, nstep),
        iprfrq=max(1, nstep),
        isvfrq=max(1, nstep),
    )
    dyn_kw["new"] = True
    dyn_kw["start"] = True
    dyn_kw["nstep"] = nstep

    if not quiet:
        print(
            f"mmml md-system dyna-probe: {nstep} NVE step(s), dt={dt_fs} fs, "
            f"PBC={use_pbc}",
            flush=True,
        )
    run_dynamics(dyn_kw)

    post = pycharmm_dyna_snapshot(
        mlpot_ctx,
        calc,
        z=z,
        n_monomers=n_monomers,
        use_pbc=use_pbc,
        L=L,
        label="post_dyna",
        step=nstep,
        quiet=quiet,
        reference_path=ref_path_resolved,
        ref_frame=ref_frame if ref_path_resolved is not None else None,
        ref_energy_unit=ref_e_unit,
        ref_force_unit=ref_f_unit,
        reference=reference,
    )
    post["ml_regime"] = classify_ml_regime(
        float(post["com_dist_A"]),
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
    )

    out_dir = Path(getattr(args, "output_dir", Path("artifacts/dyna_probe"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = getattr(args, "dyna_probe_output", None)
    if out_path is None:
        out_path = out_dir / "dyna_probe.json"
    else:
        out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "mode": "dyna_probe",
        "backend": "pycharmm",
        "composition": getattr(args, "composition", None),
        "evaluate_npz": str(ctx_prep["npz_path"]),
        "evaluate_frame": frame,
        "reference_npz": str(ref_path_resolved) if ref_path_resolved is not None else None,
        "reference_frame": ref_frame if ref_path_resolved is not None else None,
        "cutoffs": {
            "ml_switch_width_A": ml_w,
            "mm_switch_on_A": mm_on,
            "mm_switch_width_A": mm_w,
        },
        "dynamics": {
            "nstep": nstep,
            "timestep_fs": dt_fs,
            "timestep_ps": timestep_ps,
            "echeck_kcal_mol": echeck,
            "use_pbc": use_pbc,
        },
        "snapshots": [pre, post],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not quiet:
        print(f"mmml md-system dyna-probe: wrote {out_path}", flush=True)
        for snap in (pre, post):
            label = snap["label"]
            rmse = snap.get("reference_compare", {}).get("force_rmse_eV_A")
            charmm_rmse = snap.get("reference_compare", {}).get("force_rmse_charmm_total_eV_A")
            sph_rmse = snap.get("reference_compare", {}).get("force_rmse_spherical_fn_eV_A")
            print(
                f"  {label}: E={snap['energy_eV']:.6f} eV, GRMS={snap['grms_kcal_mol_A']:.4f}, "
                f"force_rmse={rmse}, spherical_fn={sph_rmse}, charmm_total={charmm_rmse}",
                flush=True,
            )
    return 0
