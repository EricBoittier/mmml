"""Single-point MMML evaluation from an NPZ geometry file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mmml.cli.run.md_handoff import (
    MdHandoffState,
    apply_handoff_geometry_to_atoms,
    cluster_geometry_from_handoff,
    ensure_psf_for_handoff_cluster,
    load_handoff_from_npz,
    resolve_handoff_box,
    set_handoff_in,
)
from mmml.interfaces.pycharmmInterface.cutoffs import handoff_widths_from_args
from mmml.interfaces.pycharmmInterface.hybrid_reference import (
    GeometryNpzPayload as EvaluateNpzPayload,
    apply_npz_charges_to_psf,
    load_geometry_npz,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

load_evaluate_npz = load_geometry_npz

HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
BOHR_TO_ANG = 0.529177210903
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG
EV_ANG_TO_HARTREE_BOHR = BOHR_TO_ANG / HARTREE_TO_EV


def _evaluate_int_arg(args: Any, name: str, default: int) -> int:
    value = getattr(args, name, default)
    if value is None:
        return int(default)
    return int(value)


def resolve_evaluate_use_pbc(args: Any, handoff: MdHandoffState) -> bool:
    if handoff.cell is not None or bool(handoff.pbc):
        return True
    setup = str(getattr(args, "setup", "free_nve"))
    if setup.startswith("pbc_"):
        return True
    if setup.startswith("free_"):
        return False
    if getattr(args, "box_size", None) is not None:
        return True
    return False


def _resolve_evaluate_backend(args: Any) -> str:
    backend = str(getattr(args, "backend", "auto"))
    if backend == "auto":
        if str(getattr(args, "setup", "")).startswith("pbc_"):
            return "jaxmd" if str(getattr(args, "setup", "")) == "pbc_npt" else "ase"
        if str(getattr(args, "setup", "")) in ("pycharmm_minimize", "pycharmm_full"):
            return "pycharmm"
        return "ase"
    return backend


def _mmml_cutoff_args(args: Any) -> tuple[float, float, float]:
    ml_w, mm_on, mm_w = handoff_widths_from_args(args)
    return ml_w, mm_on, mm_w


def _build_atoms_for_evaluate(
    *,
    z: np.ndarray,
    r0: np.ndarray,
    handoff: MdHandoffState,
    monomer_offsets: np.ndarray,
    use_pbc: bool,
    L: float | None,
) -> Any:
    from ase import Atoms

    pos = np.asarray(r0, dtype=np.float64)
    if use_pbc and L is not None:
        pos = pos - pos.mean(axis=0) + 0.5 * float(L)
    atoms = Atoms(numbers=z, positions=pos)
    if use_pbc and L is not None:
        atoms.set_cell([float(L), float(L), float(L)])
        atoms.set_pbc(True)
    else:
        atoms.set_pbc(False)
    apply_handoff_geometry_to_atoms(atoms, handoff, monomer_offsets=monomer_offsets)
    return atoms


def _evaluate_ase_mmml(
    args: Any,
    *,
    atoms: Any,
    z: np.ndarray,
    n_monomers: int,
    atoms_per_list: list[int],
    base_ckpt_dir: Path,
    use_pbc: bool,
    L: float | None,
    at_codes_override: np.ndarray | None,
) -> dict[str, Any]:
    from mmml.cli.run.md_pbc_suite.ase import _factory_mmml

    ml_w, mm_on, mm_w = _mmml_cutoff_args(args)
    atoms_per = atoms_per_list[0] if len(set(atoms_per_list)) == 1 else atoms_per_list
    calc = _factory_mmml(
        z=z,
        r=atoms.get_positions(),
        n_mol=n_monomers,
        atoms_per=atoms_per,
        base_ckpt_dir=base_ckpt_dir,
        ml_cut=ml_w,
        mm_sw=mm_on,
        mm_cut=mm_w,
        cell_scalar=float(L) if use_pbc and L is not None else None,
        verbose=bool(getattr(args, "verbose_calc", False)),
        jax_md_capacity_multiplier=float(getattr(args, "jax_md_capacity_multiplier", 1.25)),
        jax_md_capacity_growth_factor=float(
            getattr(args, "jax_md_capacity_growth_factor", 1.5)
        ),
        jax_md_max_overflow_retries=_evaluate_int_arg(args, "jax_md_max_overflow_retries", 4),
        jax_md_overflow_fallback_to_cell_list=not bool(
            getattr(args, "jax_md_disable_fallback", False)
        ),
        jax_md_update_interval=_evaluate_int_arg(args, "jax_md_update_interval", 1),
        jax_md_skin_distance=float(getattr(args, "jax_md_skin_distance", 0.2)),
        max_pairs=_evaluate_int_arg(args, "max_pairs", 20_000),
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        flat_bottom_force_const=float(getattr(args, "flat_bottom_k", 1.0)),
        flat_bottom_mode=str(getattr(args, "flat_bottom_mode", "system")),
        min_com_restraint_distance=getattr(args, "min_com_restraint_distance", None),
        min_com_restraint_force_const=float(getattr(args, "min_com_restraint_k", 1.0)),
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        ml_compute_dtype=getattr(args, "ml_compute_dtype", None),
        at_codes_override=at_codes_override,
    )
    atoms.calc = calc
    energy_eV = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return {
        "energy_eV": energy_eV,
        "energy_kcal_mol": energy_eV * float(ev2kcalmol),
        "forces_eV_A": forces.tolist(),
        "max_force_eV_A": float(np.abs(forces).max()),
        "rms_force_eV_A": float(np.sqrt(np.mean(forces**2))),
        "n_atoms": int(len(z)),
        "n_monomers": int(n_monomers),
        "pbc": bool(use_pbc),
        "box_A": float(L) if L is not None else None,
    }


def _evaluate_jaxmd_mmml(
    args: Any,
    *,
    atoms: Any,
    z: np.ndarray,
    n_monomers: int,
    atoms_per_list: list[int],
    base_ckpt_dir: Path,
    use_pbc: bool,
    L: float | None,
    at_codes_override: np.ndarray | None,
) -> dict[str, Any]:
    from mmml.interfaces.pycharmmInterface.calculator_utils import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
    import pycharmm.psf as psf

    ml_w, mm_on, mm_w = _mmml_cutoff_args(args)
    if at_codes_override is not None:
        at_codes = np.asarray(at_codes_override, dtype=int)
    else:
        at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_list,
        N_MONOMERS=n_monomers,
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max(atoms_per_list) * 2,
        cell=False if not use_pbc else float(L),
        at_codes_override=at_codes,
        max_pairs=_evaluate_int_arg(args, "max_pairs", 20_000),
        jax_md_capacity_multiplier=float(getattr(args, "jax_md_capacity_multiplier", 1.25)),
        jax_md_capacity_growth_factor=float(
            getattr(args, "jax_md_capacity_growth_factor", 1.5)
        ),
        jax_md_max_overflow_retries=_evaluate_int_arg(args, "jax_md_max_overflow_retries", 4),
        jax_md_overflow_fallback_to_cell_list=not bool(
            getattr(args, "jax_md_disable_fallback", False)
        ),
        jax_md_update_interval=_evaluate_int_arg(args, "jax_md_update_interval", 1),
        jax_md_skin_distance=float(getattr(args, "jax_md_skin_distance", 0.2)),
        flat_bottom_radius=getattr(args, "flat_bottom_radius", None),
        flat_bottom_force_const=float(getattr(args, "flat_bottom_k", 1.0)),
        flat_bottom_mode=str(getattr(args, "flat_bottom_mode", "system")),
        min_com_restraint_distance=getattr(args, "min_com_restraint_distance", None),
        min_com_restraint_force_const=float(getattr(args, "min_com_restraint_k", 1.0)),
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        ml_compute_dtype=getattr(args, "ml_compute_dtype", None),
    )
    cutoff = CutoffParameters(ml_switch_width=ml_w, mm_switch_on=mm_on, mm_switch_width=mm_w)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=atoms.get_positions(),
        n_monomers=n_monomers,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=False,
    )
    if len(calc_result) == 3:
        calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None

    atoms.calc = calc
    energy_eV = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    ase_result = {
        "energy_eV": energy_eV,
        "energy_kcal_mol": energy_eV * float(ev2kcalmol),
        "forces_eV_A": forces.tolist(),
        "max_force_eV_A": float(np.abs(forces).max()),
        "rms_force_eV_A": float(np.sqrt(np.mean(forces**2))),
        "n_atoms": int(len(z)),
        "n_monomers": int(n_monomers),
        "pbc": bool(use_pbc),
        "box_A": float(L) if L is not None else None,
        "path": "ase_interface",
    }

    if spherical_cutoff_calculator is None:
        return ase_result

    import jax.numpy as jnp

    pos = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
    z_j = jnp.asarray(z, dtype=jnp.int32)
    mm_pair_idx = None
    mm_pair_mask = None
    if get_update_fn is not None and use_pbc and L is not None:
        box_nl = np.array([float(L), float(L), float(L)], dtype=np.float64)
        pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
        update_fn = get_update_fn(pos_np, cutoff, box=box_nl)
        if update_fn is not None:
            mm_pair_idx, mm_pair_mask = update_fn(pos_np, box=box_nl)
    box_j = None
    if use_pbc and L is not None:
        box_j = jnp.array([float(L), float(L), float(L)], dtype=jnp.float32)
    out = spherical_cutoff_calculator(
        atomic_numbers=z_j,
        positions=pos,
        n_monomers=n_monomers,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        mm_pair_idx=mm_pair_idx,
        mm_pair_mask=mm_pair_mask,
        box=box_j,
    )
    energy_jit = float(np.asarray(out.energy).reshape(-1)[0])
    forces_jit = np.asarray(out.forces, dtype=np.float64)
    ase_result["jaxmd_jit"] = {
        "energy_eV": energy_jit,
        "energy_kcal_mol": energy_jit * float(ev2kcalmol),
        "max_force_eV_A": float(np.abs(forces_jit).max()),
        "rms_force_eV_A": float(np.sqrt(np.mean(forces_jit**2))),
    }
    return ase_result


def _evaluate_pycharmm(
    args: Any,
    *,
    z: np.ndarray,
    positions: np.ndarray,
    n_monomers: int,
    use_pbc: bool,
    L: float | None,
) -> dict[str, Any]:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_charmm_output_from_args,
        charmm_energy_row,
        refresh_mlpot_energy_and_grms,
        resolve_pbc_box_side,
        resolve_use_pbc,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import _register_mlpot_context
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        setup_default_nbonds,
        sync_charmm_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint

    apply_charmm_output_from_args(args)
    args.free_space = not use_pbc
    if use_pbc:
        box_side = float(L) if L is not None else resolve_pbc_box_side(args, positions)
        setup_charmm_environment(use_pbc=True, cubic_box_side_A=box_side)
    else:
        setup_default_nbonds()

    ckpt = resolve_checkpoint(getattr(args, "checkpoint", None))
    ctx, calc = _register_mlpot_context(
        z,
        positions,
        ckpt,
        len(z),
        n_monomers,
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_gpu_count=getattr(args, "ml_gpu_count", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        cubic_box_side_A=float(L) if use_pbc and L is not None else None,
        verbose=not bool(getattr(args, "quiet", False)),
        args=args,
    )
    sync_charmm_positions(np.asarray(positions, dtype=np.float64))
    grms = refresh_mlpot_energy_and_grms(ctx, context="evaluate-npz")
    charmm_row = charmm_energy_row()
    total_kcal = float(charmm_row.get("ENER", charmm_row.get("ENERGY", 0.0)))
    return {
        "energy_kcal_mol": total_kcal,
        "energy_eV": total_kcal / float(ev2kcalmol),
        "grms_kcal_mol_A": float(grms),
        "charmm_energy_terms_kcal_mol": charmm_row,
        "n_atoms": int(len(z)),
        "n_monomers": int(n_monomers),
        "pbc": bool(use_pbc),
        "box_A": float(L) if L is not None else None,
        "path": "pycharmm_mlpot_callback",
    }


def run_evaluate_npz(args: Any) -> int:
    """Evaluate MMML energy/forces at NPZ geometry in the selected backend runtime."""
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import _cubic_box_length, _parse_composition

    npz_path = Path(args.evaluate_npz).expanduser().resolve()
    frame = int(getattr(args, "evaluate_frame", 0) or 0)
    payload = load_evaluate_npz(npz_path, frame=frame)
    handoff = payload.handoff
    set_handoff_in(handoff)

    z, r0, atoms_per_list, residue_labels, composition_summary = (
        cluster_geometry_from_handoff(
            handoff,
            composition=getattr(args, "composition", None),
            n_molecules=int(getattr(args, "n_molecules", 1) or 1),
        )
    )
    n_monomers = len(atoms_per_list)
    monomer_offsets = np.zeros(n_monomers + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))

    if getattr(args, "composition", None):
        handoff_composition = _parse_composition(str(args.composition))
    elif composition_summary:
        handoff_composition = [
            (str(res), int(cnt)) for res, cnt in composition_summary.items()
        ]
    else:
        handoff_composition = [(residue_labels[0], n_monomers)]

    ensure_psf_for_handoff_cluster(
        composition=handoff_composition,
        atomic_numbers=z,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
        positions=r0,
        quiet=bool(getattr(args, "quiet", False)),
    )
    if payload.charges is not None:
        apply_npz_charges_to_psf(payload.charges)

    use_pbc = resolve_evaluate_use_pbc(args, handoff)
    auto_L = float(_cubic_box_length(r0, float(getattr(args, "ml_cutoff", 0.1))))
    L_resolved, box_source, box_warnings = resolve_handoff_box(
        handoff,
        yaml_box_size=getattr(args, "box_size", None),
        free_space=not use_pbc,
        auto_box_from_geometry=auto_L,
        require_cell=bool(getattr(args, "handoff_require_cell", False)),
    )
    L = float(L_resolved) if L_resolved is not None else (auto_L if use_pbc else None)

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        base_ckpt_dir, _ = resolve_checkpoint_paths(
            Path(args.checkpoint).expanduser().resolve()
        )

    atoms = _build_atoms_for_evaluate(
        z=z,
        r0=r0,
        handoff=handoff,
        monomer_offsets=monomer_offsets,
        use_pbc=use_pbc,
        L=L,
    )

    backend = _resolve_evaluate_backend(args)
    if backend == "pycharmm":
        from mmml.interfaces.pycharmmInterface.charmm_mpi import prepare_serial_charmm_mpi_env
        from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
            apply_jax_compile_xla_flags,
        )
        from mmml.interfaces.pycharmmInterface.jax_device_policy import (
            apply_mlpot_jax_compilation_cache_env,
        )

        prepare_serial_charmm_mpi_env()
        apply_mlpot_jax_compilation_cache_env(quiet=True)
        apply_jax_compile_xla_flags(quiet=True)

    if backend == "pycharmm":
        metrics = _evaluate_pycharmm(
            args,
            z=z,
            positions=atoms.get_positions(),
            n_monomers=n_monomers,
            use_pbc=use_pbc,
            L=L,
        )
    elif backend == "jaxmd":
        metrics = _evaluate_jaxmd_mmml(
            args,
            atoms=atoms,
            z=z,
            n_monomers=n_monomers,
            atoms_per_list=atoms_per_list,
            base_ckpt_dir=base_ckpt_dir,
            use_pbc=use_pbc,
            L=L,
            at_codes_override=payload.at_codes,
        )
    else:
        metrics = _evaluate_ase_mmml(
            args,
            atoms=atoms,
            z=z,
            n_monomers=n_monomers,
            atoms_per_list=atoms_per_list,
            base_ckpt_dir=base_ckpt_dir,
            use_pbc=use_pbc,
            L=L,
            at_codes_override=payload.at_codes,
        )

    out_dir = Path(getattr(args, "output_dir", Path("artifacts/md_evaluate"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = getattr(args, "evaluate_output", None)
    if out_path is None:
        out_path = out_dir / "evaluate.json"
    else:
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "backend": backend,
        "setup": str(getattr(args, "setup", "")),
        "npz": str(npz_path),
        "composition": getattr(args, "composition", None),
        "checkpoint": str(base_ckpt_dir),
        "box_source": box_source,
        "box_warnings": box_warnings,
        "mm_params_from_npz": {
            "charges": payload.charges is not None,
            "at_codes": payload.at_codes is not None,
            "epsilon": payload.epsilon is not None,
            "sigma": payload.sigma is not None,
        },
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(f"mmml md-system evaluate-npz ({backend}):", flush=True)
        print(f"  energy = {metrics.get('energy_eV', metrics.get('energy_kcal_mol'))}", flush=True)
        print(f"  max|F| = {metrics.get('max_force_eV_A', metrics.get('grms_kcal_mol_A'))}", flush=True)
        print(f"  wrote {out_path}", flush=True)
    return 0
