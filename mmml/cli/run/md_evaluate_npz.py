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


def energy_to_ev(values: np.ndarray | float, unit: str) -> np.ndarray | float:
    unit_l = str(unit).lower()
    arr = np.asarray(values, dtype=np.float64)
    scalar = arr.ndim == 0
    if unit_l in {"ev"}:
        out = arr
    elif unit_l in {"hartree", "ha"}:
        out = arr * HARTREE_TO_EV
    elif unit_l in {"kcal", "kcal/mol", "kcal_mol"}:
        out = arr * (1.0 / float(ev2kcalmol))
    else:
        raise ValueError(f"Unsupported energy unit: {unit}")
    return float(out) if scalar else out


def forces_hartree_bohr_to_ev_ang(forces: np.ndarray) -> np.ndarray:
    return np.asarray(forces, dtype=np.float64) * HARTREE_BOHR_TO_EV_ANG


def forces_ev_ang_to_hartree_bohr(forces: np.ndarray) -> np.ndarray:
    return np.asarray(forces, dtype=np.float64) * EV_ANG_TO_HARTREE_BOHR


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:
        import itertools

        n_rows, n_cols = cost.shape
        if n_rows != n_cols or n_rows > 9:
            raise RuntimeError(
                "scipy is required for reference atom reordering when n_atoms > 9"
            ) from None
        best_perm: tuple[int, ...] | None = None
        best_cost = float("inf")
        for perm in itertools.permutations(range(n_cols)):
            value = float(sum(cost[i, perm[i]] for i in range(n_rows)))
            if value < best_cost:
                best_cost = value
                best_perm = perm
        assert best_perm is not None
        return np.arange(n_rows), np.asarray(best_perm, dtype=int)


def _assign_atoms_by_element(
    source_positions: np.ndarray,
    source_numbers: np.ndarray,
    target_positions: np.ndarray,
    target_numbers: np.ndarray,
) -> np.ndarray:
    if sorted(source_numbers.tolist()) != sorted(target_numbers.tolist()):
        raise ValueError("Reference/evaluate atomic-number multisets differ")
    permutation = np.empty(int(len(target_numbers)), dtype=int)
    for atomic_number in sorted(set(int(z) for z in target_numbers.tolist())):
        target_idx = np.where(target_numbers == atomic_number)[0]
        source_idx = np.where(source_numbers == atomic_number)[0]
        diff = target_positions[target_idx, None, :] - source_positions[source_idx][None, :, :]
        cost = np.linalg.norm(diff, axis=2)
        rows, cols = _linear_sum_assignment(cost)
        permutation[target_idx[rows]] = source_idx[cols]
    return permutation


def align_reference_frame_to_evaluate(
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    evaluate_positions: np.ndarray,
    evaluate_numbers: np.ndarray,
    *,
    reference_forces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Reorder reference rows to match evaluate atom order (geometry + element matching)."""
    ref_r = np.asarray(reference_positions, dtype=np.float64).reshape(-1, 3)
    ref_z = np.asarray(reference_numbers, dtype=np.int32).reshape(-1)
    eval_r = np.asarray(evaluate_positions, dtype=np.float64).reshape(-1, 3)
    eval_z = np.asarray(evaluate_numbers, dtype=np.int32).reshape(-1)
    if ref_r.shape[0] != eval_r.shape[0]:
        raise ValueError(
            f"Reference has {ref_r.shape[0]} atoms but evaluate geometry has {eval_r.shape[0]}"
        )
    meta: dict[str, Any] = {"reference_reordered": False}
    if np.array_equal(ref_z, eval_z):
        ref_f = (
            None
            if reference_forces is None
            else np.asarray(reference_forces, dtype=np.float64).reshape(-1, 3)
        )
        return ref_r, ref_f, meta

    perm = _assign_atoms_by_element(ref_r, ref_z, eval_r, eval_z)
    ref_r_aligned = ref_r[perm]
    ref_f_aligned = None
    if reference_forces is not None:
        ref_f_aligned = np.asarray(reference_forces, dtype=np.float64).reshape(-1, 3)[perm]
    meta["reference_reordered"] = True
    meta["reference_permutation"] = perm.tolist()
    meta["position_rmsd_after_reorder_A"] = float(
        np.sqrt(np.mean((ref_r_aligned - eval_r) ** 2))
    )
    return ref_r_aligned, ref_f_aligned, meta


def _forces_from_metrics(metrics: dict[str, Any]) -> np.ndarray | None:
    if "forces_eV_A" in metrics:
        return np.asarray(metrics["forces_eV_A"], dtype=np.float64)
    return None


def _energy_ev_from_metrics(metrics: dict[str, Any]) -> float | None:
    if "energy_eV" in metrics:
        return float(metrics["energy_eV"])
    if "energy_kcal_mol" in metrics:
        return float(metrics["energy_kcal_mol"]) / float(ev2kcalmol)
    return None


def save_evaluate_trajectory_npz(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy_eV: float,
    forces_eV_A: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a single-frame trajectory-style NPZ (R/F/E/Z/N) for downstream tools."""
    z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    r = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n_atoms = int(len(z))
    payload: dict[str, Any] = {
        "N": np.array([n_atoms], dtype=np.int32),
        "Z": z.reshape(1, n_atoms),
        "R": r.reshape(1, n_atoms, 3),
        "E": np.array([float(energy_eV) * EV_TO_HARTREE], dtype=np.float64),
        "E_eV": np.array([float(energy_eV)], dtype=np.float64),
    }
    if forces_eV_A is not None:
        f_ev = np.asarray(forces_eV_A, dtype=np.float64).reshape(n_atoms, 3)
        payload["F"] = f_ev.reshape(1, n_atoms, 3)
        payload["F_hartree_bohr"] = forces_ev_ang_to_hartree_bohr(f_ev).reshape(1, n_atoms, 3)
    if metadata:
        payload["metadata"] = json.dumps(metadata)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_evaluate_extxyz(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy_eV: float,
    forces_eV_A: np.ndarray | None,
) -> None:
    """Write one extended-XYZ frame with energy/forces attached for ASE/Ovito GUIs."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    atoms = Atoms(
        numbers=np.asarray(atomic_numbers, dtype=int),
        positions=np.asarray(positions, dtype=np.float64),
    )
    calc_kwargs: dict[str, Any] = {"energy": float(energy_eV)}
    if forces_eV_A is not None:
        calc_kwargs["forces"] = np.asarray(forces_eV_A, dtype=np.float64)
    atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="extxyz")


def save_evaluate_trajectory_npz_multi(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energies_eV: np.ndarray,
    forces_eV_A: np.ndarray | None,
    frame_indices: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write multi-frame trajectory NPZ (R/F/E/Z/N)."""
    z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    r = np.asarray(positions, dtype=np.float64)
    if r.ndim != 3:
        raise ValueError(f"positions must be (n_frames, n_atoms, 3), got {r.shape}")
    n_frames, n_atoms, _ = r.shape
    if int(len(z)) != int(n_atoms):
        raise ValueError(f"atomic_numbers length {len(z)} != n_atoms {n_atoms}")
    e_ev = np.asarray(energies_eV, dtype=np.float64).reshape(-1)
    if int(e_ev.shape[0]) != int(n_frames):
        raise ValueError(f"energies length {e_ev.shape[0]} != n_frames {n_frames}")
    payload: dict[str, Any] = {
        "N": np.full(n_frames, n_atoms, dtype=np.int32),
        "Z": np.broadcast_to(z.reshape(1, n_atoms), (n_frames, n_atoms)).copy(),
        "R": r,
        "E": e_ev * EV_TO_HARTREE,
        "E_eV": e_ev,
    }
    if frame_indices is not None:
        payload["source_indices"] = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if forces_eV_A is not None:
        f_ev = np.asarray(forces_eV_A, dtype=np.float64)
        if f_ev.shape != (n_frames, n_atoms, 3):
            raise ValueError(f"forces shape {f_ev.shape} != ({n_frames}, {n_atoms}, 3)")
        payload["F"] = f_ev
        payload["F_hartree_bohr"] = forces_ev_ang_to_hartree_bohr(f_ev.reshape(-1, 3)).reshape(
            n_frames, n_atoms, 3
        )
    if metadata:
        payload["metadata"] = json.dumps(metadata)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_evaluate_extxyz_multi(
    path: Path,
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energies_eV: np.ndarray,
    forces_eV_A: np.ndarray | None,
) -> None:
    """Write multi-frame extended XYZ with attached energy/forces."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    r = np.asarray(positions, dtype=np.float64)
    e_ev = np.asarray(energies_eV, dtype=np.float64).reshape(-1)
    n_frames = int(r.shape[0])
    frames: list[Any] = []
    for i in range(n_frames):
        atoms = Atoms(numbers=z, positions=r[i])
        calc_kwargs: dict[str, Any] = {"energy": float(e_ev[i])}
        if forces_eV_A is not None:
            calc_kwargs["forces"] = np.asarray(forces_eV_A[i], dtype=np.float64)
        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
        frames.append(atoms)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), frames, format="extxyz")


def resolve_evaluate_max_frames(args: Any) -> int:
    """Frame cap for --evaluate-npz batch runs against --evaluate-reference-npz."""
    raw = getattr(args, "max_frames", None)
    if raw is None:
        return 1
    return int(raw)


def should_evaluate_reference_trajectory(args: Any) -> bool:
    """True when MMML should loop over frames from --evaluate-reference-npz."""
    if getattr(args, "evaluate_reference_npz", None) is None:
        return False
    return resolve_evaluate_max_frames(args) != 1


def compare_evaluate_to_reference_npz(
    reference_path: Path,
    *,
    frame: int,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    energy_eV: float,
    forces_eV_A: np.ndarray | None,
    reference_energy_unit: str = "hartree",
    reference_force_unit: str = "hartree_bohr",
) -> dict[str, Any]:
    """Compare MMML single-point results to a reference trajectory NPZ frame."""
    ref_path = Path(reference_path).expanduser().resolve()
    with np.load(ref_path, allow_pickle=True) as ref:
        if "R" not in ref.files:
            raise ValueError(f"{ref_path.name} has no 'R' key; expected trajectory NPZ")
        n_atoms = int(len(atomic_numbers))
        ref_frame = int(frame)
        if "N" in ref.files:
            ref_n = int(np.asarray(ref["N"], dtype=int).reshape(-1)[ref_frame])
        else:
            ref_n = int(np.asarray(ref["R"], dtype=np.float64).shape[1])
        ref_z = np.asarray(ref["Z"][ref_frame, :ref_n], dtype=np.int32)
        ref_r = np.asarray(ref["R"][ref_frame, :ref_n], dtype=np.float64)
        if ref_n != n_atoms:
            raise ValueError(
                f"Reference frame has N={ref_n} atoms but evaluate geometry has {n_atoms}"
            )
        eval_z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
        eval_r = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        ref_f_raw = (
            np.asarray(ref["F"][ref_frame, :ref_n], dtype=np.float64)
            if forces_eV_A is not None and "F" in ref.files
            else None
        )
        ref_r, ref_f_raw, align_meta = align_reference_frame_to_evaluate(
            ref_r,
            ref_z,
            eval_r,
            eval_z,
            reference_forces=ref_f_raw,
        )
        pos_rmsd = float(np.sqrt(np.mean((ref_r - eval_r) ** 2)))
        out: dict[str, Any] = {
            "reference_npz": str(ref_path),
            "reference_frame": ref_frame,
            "reference_energy_unit": reference_energy_unit,
            "reference_force_unit": reference_force_unit,
            "position_rmsd_A": pos_rmsd,
            "n_atoms": n_atoms,
            **align_meta,
        }
        if "E" in ref.files:
            ref_e_raw = float(np.asarray(ref["E"], dtype=np.float64).reshape(-1)[ref_frame])
            ref_e_ev = float(energy_to_ev(ref_e_raw, reference_energy_unit))
            delta_e = float(energy_eV) - ref_e_ev
            out["reference_energy_raw"] = ref_e_raw
            out["reference_energy_eV"] = ref_e_ev
            out["predicted_energy_eV"] = float(energy_eV)
            out["delta_energy_eV"] = delta_e
            out["abs_delta_energy_eV"] = abs(delta_e)
        if forces_eV_A is not None and ref_f_raw is not None:
            unit_l = reference_force_unit.lower()
            if unit_l in {"hartree_bohr", "hartree/bohr", "ha/bohr"}:
                ref_f_ev = forces_hartree_bohr_to_ev_ang(ref_f_raw)
            elif unit_l in {"ev_ang", "ev/a", "ev/ang", "ev/angstrom"}:
                ref_f_ev = ref_f_raw
            else:
                raise ValueError(f"Unsupported reference force unit: {reference_force_unit}")
            pred_f = np.asarray(forces_eV_A, dtype=np.float64).reshape(ref_n, 3)
            delta_f = pred_f - ref_f_ev
            out["force_rmse_eV_A"] = float(np.sqrt(np.mean(delta_f**2)))
            out["force_mae_eV_A"] = float(np.mean(np.abs(delta_f)))
            out["force_max_abs_eV_A"] = float(np.max(np.abs(delta_f)))
        if "source_indices" in ref.files:
            out["reference_source_index"] = int(
                np.asarray(ref["source_indices"], dtype=int).reshape(-1)[ref_frame]
            )
    return out


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


def _attach_ase_mmml_calculator(
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
) -> Any:
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
    return calc


def _evaluate_atoms_mmml(
    atoms: Any,
    *,
    z: np.ndarray,
    n_monomers: int,
    use_pbc: bool,
    L: float | None,
) -> dict[str, Any]:
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
    _attach_ase_mmml_calculator(
        args,
        atoms=atoms,
        z=z,
        n_monomers=n_monomers,
        atoms_per_list=atoms_per_list,
        base_ckpt_dir=base_ckpt_dir,
        use_pbc=use_pbc,
        L=L,
        at_codes_override=at_codes_override,
    )
    return _evaluate_atoms_mmml(
        atoms,
        z=z,
        n_monomers=n_monomers,
        use_pbc=use_pbc,
        L=L,
    )


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


def _prepare_evaluate_npz_context(args: Any) -> dict[str, Any]:
    """Shared setup for single- and multi-frame ``--evaluate-npz`` runs."""
    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_pbc_suite.ase import _cubic_box_length, _parse_composition
    from mmml.interfaces.pycharmmInterface.hybrid_reference import load_reference_trajectory_npz

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

    reference = None
    ref_path = getattr(args, "evaluate_reference_npz", None)
    if ref_path is not None and should_evaluate_reference_trajectory(args):
        ref_path = Path(ref_path).expanduser().resolve()
        n_atoms_monomer = int(len(z) // n_monomers)
        reference = load_reference_trajectory_npz(
            ref_path,
            z_fallback=z,
            n_atoms_monomer=n_atoms_monomer,
            n_monomers=n_monomers,
            max_frames=resolve_evaluate_max_frames(args),
        )
        r0 = np.asarray(reference.R[int(reference.frame_indices[0])], dtype=np.float64)

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

    if reference is not None:
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
    else:
        atoms = _build_atoms_for_evaluate(
            z=z,
            r0=r0,
            handoff=handoff,
            monomer_offsets=monomer_offsets,
            use_pbc=use_pbc,
            L=L,
        )

    return {
        "npz_path": npz_path,
        "frame": frame,
        "payload": payload,
        "handoff": handoff,
        "z": z,
        "atoms_per_list": atoms_per_list,
        "residue_labels": residue_labels,
        "n_monomers": n_monomers,
        "use_pbc": use_pbc,
        "L": L,
        "box_source": box_source,
        "box_warnings": box_warnings,
        "base_ckpt_dir": base_ckpt_dir,
        "atoms": atoms,
        "reference": reference,
    }


def _evaluate_reference_trajectory(args: Any, ctx: dict[str, Any]) -> int:
    """Evaluate MMML on multiple frames from ``--evaluate-reference-npz``."""
    from mmml.interfaces.pycharmmInterface.charmm_mpi import prepare_serial_charmm_mpi_env
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import apply_jax_compile_xla_flags
    from mmml.interfaces.pycharmmInterface.jax_device_policy import (
        apply_mlpot_jax_compilation_cache_env,
    )

    reference = ctx["reference"]
    assert reference is not None
    backend = _resolve_evaluate_backend(args)
    if backend not in {"ase", "jaxmd", "pycharmm"}:
        raise ValueError(
            f"Reference trajectory evaluation supports ase/jaxmd/pycharmm backends, got {backend!r}"
        )
    if backend == "pycharmm":
        prepare_serial_charmm_mpi_env()
        apply_mlpot_jax_compilation_cache_env(quiet=True)
        apply_jax_compile_xla_flags(quiet=True)

    z = ctx["z"]
    atoms = ctx["atoms"]
    n_monomers = ctx["n_monomers"]
    atoms_per_list = ctx["atoms_per_list"]
    base_ckpt_dir = ctx["base_ckpt_dir"]
    use_pbc = ctx["use_pbc"]
    L = ctx["L"]
    payload = ctx["payload"]
    frame_indices = np.asarray(reference.frame_indices, dtype=int).reshape(-1)
    n_eval = int(len(frame_indices))

    if backend == "ase":
        _attach_ase_mmml_calculator(
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
    elif backend == "jaxmd":
        _evaluate_jaxmd_mmml(
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
        _evaluate_pycharmm(
            args,
            z=z,
            positions=atoms.get_positions(),
            n_monomers=n_monomers,
            use_pbc=use_pbc,
            L=L,
        )

    energies: list[float] = []
    forces_list: list[np.ndarray] = []
    positions_out: list[np.ndarray] = []
    per_frame_compare: list[dict[str, Any]] = []
    ref_energy_unit = str(getattr(args, "evaluate_reference_energy_unit", "hartree"))
    ref_force_unit = str(getattr(args, "evaluate_reference_force_unit", "hartree_bohr"))

    if not getattr(args, "quiet", False):
        print(
            f"mmml md-system evaluate-npz ({backend}): "
            f"evaluating {n_eval} frames from {reference.path.name}",
            flush=True,
        )

    for ref_frame in frame_indices:
        pos = np.asarray(reference.R[int(ref_frame)], dtype=np.float64)
        atoms.set_positions(pos)
        if backend == "pycharmm":
            metrics = _evaluate_pycharmm(
                args,
                z=z,
                positions=pos,
                n_monomers=n_monomers,
                use_pbc=use_pbc,
                L=L,
            )
        else:
            metrics = _evaluate_atoms_mmml(
                atoms,
                z=z,
                n_monomers=n_monomers,
                use_pbc=use_pbc,
                L=L,
            )
        energy_eV = float(metrics["energy_eV"])
        forces_raw = metrics.get("forces_eV_A")
        if forces_raw is not None:
            forces = np.asarray(forces_raw, dtype=np.float64).reshape(-1, 3)
        else:
            forces = np.empty((0, 3))
        energies.append(energy_eV)
        positions_out.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        if forces.shape == (len(z), 3):
            forces_list.append(forces)
        try:
            cmp = compare_evaluate_to_reference_npz(
                reference.path,
                frame=int(ref_frame),
                atomic_numbers=z,
                positions=positions_out[-1],
                energy_eV=energy_eV,
                forces_eV_A=forces if forces.shape == (len(z), 3) else None,
                reference_energy_unit=ref_energy_unit,
                reference_force_unit=ref_force_unit,
            )
            cmp["status"] = "ok"
        except Exception as exc:
            cmp = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "reference_frame": int(ref_frame),
            }
        per_frame_compare.append(cmp)

    energies_arr = np.asarray(energies, dtype=np.float64)
    forces_arr = np.stack(forces_list, axis=0) if forces_list else None
    positions_arr = np.stack(positions_out, axis=0)

    out_dir = Path(getattr(args, "output_dir", Path("artifacts/md_evaluate"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = getattr(args, "evaluate_output", None)
    if out_path is None:
        out_path = out_dir / "evaluate.json"
    else:
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    ok_compares = [c for c in per_frame_compare if c.get("status") == "ok"]
    compare_summary: dict[str, Any] = {
        "status": "ok" if ok_compares else "error",
        "reference_npz": str(reference.path),
        "n_frames": n_eval,
        "frame_indices": frame_indices.tolist(),
        "per_frame": per_frame_compare,
    }
    if ok_compares and "delta_energy_eV" in ok_compares[0]:
        delta_e = np.asarray([c["delta_energy_eV"] for c in ok_compares], dtype=np.float64)
        compare_summary["mean_delta_energy_eV"] = float(delta_e.mean())
        compare_summary["rmse_delta_energy_eV"] = float(np.sqrt(np.mean(delta_e**2)))
        compare_summary["max_abs_delta_energy_eV"] = float(np.max(np.abs(delta_e)))
    if ok_compares and "force_rmse_eV_A" in ok_compares[0]:
        force_rmse = np.asarray([c["force_rmse_eV_A"] for c in ok_compares], dtype=np.float64)
        compare_summary["mean_force_rmse_eV_A"] = float(force_rmse.mean())

    result: dict[str, Any] = {
        "backend": backend,
        "setup": str(getattr(args, "setup", "")),
        "npz": str(ctx["npz_path"]),
        "composition": getattr(args, "composition", None),
        "checkpoint": str(base_ckpt_dir),
        "box_source": ctx["box_source"],
        "box_warnings": ctx["box_warnings"],
        "n_eval_frames": n_eval,
        "reference_compare": compare_summary,
        "mm_params_from_npz": {
            "charges": payload.charges is not None,
            "at_codes": payload.at_codes is not None,
            "epsilon": payload.epsilon is not None,
            "sigma": payload.sigma is not None,
        },
        "metrics": {
            "energy_eV": energies_arr.tolist(),
            "max_force_eV_A": [
                float(np.abs(f).max()) for f in (forces_list or [np.empty((len(z), 3))])
            ],
        },
    }

    artifacts: dict[str, str] = {}
    if not bool(getattr(args, "no_evaluate_save_artifacts", False)):
        npz_out = getattr(args, "evaluate_forces_npz", None) or out_dir / "evaluate.npz"
        npz_out = Path(npz_out).expanduser()
        save_evaluate_trajectory_npz_multi(
            npz_out,
            atomic_numbers=z,
            positions=positions_arr,
            energies_eV=energies_arr,
            forces_eV_A=forces_arr,
            frame_indices=frame_indices,
            metadata={
                "backend": backend,
                "composition": getattr(args, "composition", None),
                "source_npz": str(ctx["npz_path"]),
                "reference_npz": str(reference.path),
            },
        )
        artifacts["npz"] = str(npz_out)

        traj_out = getattr(args, "evaluate_traj", None) or out_dir / "evaluate.extxyz"
        traj_out = Path(traj_out).expanduser()
        save_evaluate_extxyz_multi(
            traj_out,
            atomic_numbers=z,
            positions=positions_arr,
            energies_eV=energies_arr,
            forces_eV_A=forces_arr,
        )
        artifacts["extxyz"] = str(traj_out)

        compare_path = getattr(args, "evaluate_compare_output", None) or out_dir / "evaluate_compare.json"
        compare_path = Path(compare_path).expanduser()
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")
        artifacts["compare_json"] = str(compare_path)

    if artifacts:
        result["artifacts"] = artifacts
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not getattr(args, "quiet", False):
        print(f"  wrote {n_eval} frames to trajectory", flush=True)
        print(f"  wrote {out_path}", flush=True)
        for label, path in artifacts.items():
            print(f"  {label} = {path}", flush=True)
        if compare_summary.get("mean_delta_energy_eV") is not None:
            print(
                f"  vs reference: mean dE = {compare_summary['mean_delta_energy_eV']:.6f} eV "
                f"(RMSE = {compare_summary['rmse_delta_energy_eV']:.6f} eV)",
                flush=True,
            )
        if compare_summary.get("mean_force_rmse_eV_A") is not None:
            print(
                f"  vs reference: mean force RMSE = {compare_summary['mean_force_rmse_eV_A']:.6f} eV/A",
                flush=True,
            )
    return 0


def run_evaluate_npz(args: Any) -> int:
    """Evaluate MMML energy/forces at NPZ geometry in the selected backend runtime."""
    ctx = _prepare_evaluate_npz_context(args)
    if ctx["reference"] is not None:
        return _evaluate_reference_trajectory(args, ctx)

    npz_path = ctx["npz_path"]
    frame = ctx["frame"]
    payload = ctx["payload"]
    z = ctx["z"]
    atoms_per_list = ctx["atoms_per_list"]
    n_monomers = ctx["n_monomers"]
    use_pbc = ctx["use_pbc"]
    L = ctx["L"]
    box_source = ctx["box_source"]
    box_warnings = ctx["box_warnings"]
    base_ckpt_dir = ctx["base_ckpt_dir"]
    atoms = ctx["atoms"]

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

    pos_out = np.asarray(atoms.get_positions(), dtype=np.float64)
    energy_eV = _energy_ev_from_metrics(metrics)
    forces_eV_A = _forces_from_metrics(metrics)
    artifacts: dict[str, str] = {}

    if not bool(getattr(args, "no_evaluate_save_artifacts", False)) and energy_eV is not None:
        npz_out = getattr(args, "evaluate_forces_npz", None)
        if npz_out is None:
            npz_out = out_dir / "evaluate.npz"
        else:
            npz_out = Path(npz_out).expanduser()
        save_evaluate_trajectory_npz(
            npz_out,
            atomic_numbers=z,
            positions=pos_out,
            energy_eV=energy_eV,
            forces_eV_A=forces_eV_A,
            metadata={
                "backend": backend,
                "composition": getattr(args, "composition", None),
                "source_npz": str(npz_path),
                "frame": frame,
            },
        )
        artifacts["npz"] = str(npz_out)

        traj_out = getattr(args, "evaluate_traj", None)
        if traj_out is None:
            traj_out = out_dir / "evaluate.extxyz"
        else:
            traj_out = Path(traj_out).expanduser()
        save_evaluate_extxyz(
            traj_out,
            atomic_numbers=z,
            positions=pos_out,
            energy_eV=energy_eV,
            forces_eV_A=forces_eV_A,
        )
        artifacts["extxyz"] = str(traj_out)

    ref_path = getattr(args, "evaluate_reference_npz", None)
    if ref_path is None:
        ref_path = getattr(args, "reference_npz", None)
    if ref_path is not None and energy_eV is not None:
        ref_frame = int(getattr(args, "evaluate_reference_frame", frame) or frame)
        compare_path = getattr(args, "evaluate_compare_output", None)
        if compare_path is None:
            compare_path = out_dir / "evaluate_compare.json"
        else:
            compare_path = Path(compare_path).expanduser()
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            compare = compare_evaluate_to_reference_npz(
                Path(ref_path),
                frame=ref_frame,
                atomic_numbers=z,
                positions=pos_out,
                energy_eV=energy_eV,
                forces_eV_A=forces_eV_A,
                reference_energy_unit=str(
                    getattr(args, "evaluate_reference_energy_unit", "hartree")
                ),
                reference_force_unit=str(
                    getattr(args, "evaluate_reference_force_unit", "hartree_bohr")
                ),
            )
            compare["status"] = "ok"
        except Exception as exc:
            compare = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "reference_npz": str(Path(ref_path).expanduser().resolve()),
                "reference_frame": ref_frame,
            }
            if not getattr(args, "quiet", False):
                print(
                    f"mmml md-system: evaluate reference compare failed: {exc}",
                    file=__import__("sys").stderr,
                    flush=True,
                )
        compare_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
        result["reference_compare"] = compare
        artifacts["compare_json"] = str(compare_path)

    if artifacts:
        result["artifacts"] = artifacts

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(f"mmml md-system evaluate-npz ({backend}):", flush=True)
        print(f"  energy = {metrics.get('energy_eV', metrics.get('energy_kcal_mol'))}", flush=True)
        print(f"  max|F| = {metrics.get('max_force_eV_A', metrics.get('grms_kcal_mol_A'))}", flush=True)
        print(f"  wrote {out_path}", flush=True)
        for label, path in artifacts.items():
            print(f"  {label} = {path}", flush=True)
        if "reference_compare" in result:
            cmp = result["reference_compare"]
            if cmp.get("status") == "error":
                print(f"  compare error: {cmp.get('error')}", flush=True)
            elif "delta_energy_eV" in cmp:
                print(
                    f"  vs reference: dE = {cmp['delta_energy_eV']:.6f} eV "
                    f"(|dE| = {cmp['abs_delta_energy_eV']:.6f} eV)",
                    flush=True,
                )
            if "force_rmse_eV_A" in cmp:
                print(
                    f"  vs reference: force RMSE = {cmp['force_rmse_eV_A']:.6f} eV/A",
                    flush=True,
                )
    return 0
