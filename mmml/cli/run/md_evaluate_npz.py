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
from mmml.data.units import (
    CALCULATOR_UNITS,
    EV_TO_HARTREE,
    convert_energy,
    convert_forces,
    energy_to_ev,
    forces_to_ev_angstrom,
    infer_reference_energy_unit,
    infer_reference_force_unit,
    normalize_energy_unit,
    normalize_force_unit,
    reference_energy_ev_at_frame,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

load_evaluate_npz = load_geometry_npz

EVALUATE_ARTIFACT_UNITS = dict(CALCULATOR_UNITS)


def _evaluate_int_arg(args: Any, name: str, default: int) -> int:
    value = getattr(args, name, default)
    if value is None:
        return int(default)
    return int(value)


def forces_hartree_bohr_to_ev_ang(forces: np.ndarray) -> np.ndarray:
    return np.asarray(
        forces_to_ev_angstrom(forces, "hartree_bohr"),
        dtype=np.float64,
    )


def forces_ev_ang_to_hartree_bohr(forces: np.ndarray) -> np.ndarray:
    return np.asarray(
        convert_forces(forces, "ev_angstrom", "hartree_bohr"),
        dtype=np.float64,
    )


def resolve_reference_units(
    reference_path: Path | str | None,
    args: Any | None = None,
) -> tuple[str, str]:
    """Resolve reference NPZ energy/force units from manifest or CLI overrides."""
    energy_unit = "hartree"
    force_unit = "hartree_bohr"
    if reference_path is not None:
        energy_unit = infer_reference_energy_unit(reference_path, default=energy_unit)
        force_unit = infer_reference_force_unit(reference_path, default=force_unit)
    if args is not None:
        cli_e = getattr(args, "evaluate_reference_energy_unit", None)
        cli_f = getattr(args, "evaluate_reference_force_unit", None)
        if cli_e is not None:
            energy_unit = str(cli_e)
        if cli_f is not None:
            force_unit = str(cli_f)
    return energy_unit, force_unit


def _calculator_energy_unit(atoms: Any | None) -> str | None:
    if atoms is None or getattr(atoms, "calc", None) is None:
        return None
    results = getattr(atoms.calc, "results", None) or {}
    return results.get("energy_unit") or results.get("units", {}).get("energy")


def normalize_metrics_to_ev(
    metrics: dict[str, Any],
    *,
    atoms: Any | None = None,
) -> dict[str, Any]:
    """Ensure metric energy/force values are numerically in eV / eV/Å."""
    out = dict(metrics)
    raw_unit = out.get("energy_unit") or _calculator_energy_unit(atoms)
    energy = out.get("energy_eV")
    if energy is None and "energy_hartree" in out:
        energy = convert_energy(out["energy_hartree"], "hartree", "ev")
    if energy is not None and raw_unit is not None:
        try:
            if normalize_energy_unit(str(raw_unit)) == "hartree":
                energy = convert_energy(energy, "hartree", "ev")
        except ValueError:
            pass
    if energy is not None:
        out["energy_eV"] = float(energy)
        out["energy_hartree"] = float(convert_energy(energy, "ev", "hartree"))
        out["energy_kcal_mol"] = float(energy) * float(ev2kcalmol)
    forces = out.get("forces_eV_A")
    force_unit = out.get("force_unit") or (
        (getattr(atoms.calc, "results", None) or {}).get("forces_unit")
        if atoms is not None and getattr(atoms, "calc", None) is not None
        else None
    )
    if forces is not None and force_unit is not None:
        try:
            if normalize_force_unit(str(force_unit)) == "hartree_bohr":
                forces = convert_forces(forces, "hartree_bohr", "ev_angstrom")
        except ValueError:
            pass
    if forces is not None:
        f_arr = np.asarray(forces, dtype=np.float64)
        out["forces_eV_A"] = f_arr.tolist() if f_arr.ndim > 1 else f_arr
        out["max_force_eV_A"] = float(np.abs(f_arr).max())
        out["rms_force_eV_A"] = float(np.sqrt(np.mean(f_arr**2)))
    out["units"] = dict(EVALUATE_ARTIFACT_UNITS)
    return out


def _print_evaluate_npz_summary(
    *,
    backend: str,
    metrics: dict[str, Any],
    out_path: Path,
    artifacts: dict[str, str],
    result: dict[str, Any],
    quiet: bool,
    n_eval_frames: int | None = None,
) -> None:
    """Print evaluate-npz completion summary (one line when ``quiet``)."""
    energy = metrics.get("energy_eV")
    if isinstance(energy, list):
        energy = float(np.mean(energy)) if energy else None
    max_f = metrics.get("max_force_eV_A")
    if isinstance(max_f, list):
        max_f = float(np.max(max_f)) if max_f else None
    if quiet:
        parts = [f"mmml evaluate-npz ({backend}):"]
        if energy is not None:
            parts.append(f"E={float(energy):.6f} eV")
        if max_f is not None:
            parts.append(f"max|F|={float(max_f):.4f} eV/A")
        if n_eval_frames is not None:
            parts.append(f"frames={int(n_eval_frames)}")
        parts.append(f"-> {out_path}")
        print(" ".join(parts), flush=True)
        return

    print(f"mmml md-system evaluate-npz ({backend}):", flush=True)
    print(f"  energy = {energy}", flush=True)
    print(f"  max|F| = {max_f}", flush=True)
    if n_eval_frames is not None:
        print(f"  wrote {n_eval_frames} frames to trajectory", flush=True)
    print(f"  wrote {out_path}", flush=True)
    for label, path in artifacts.items():
        print(f"  {label} = {path}", flush=True)
    if "reference_compare" in result:
        cmp = result["reference_compare"]
        ref_units = result.get("reference_units")
        if ref_units:
            print(
                f"  reference units: E={ref_units['energy']}, F={ref_units['force']}",
                flush=True,
            )
        if cmp.get("status") == "error":
            print(f"  compare error: {cmp.get('error')}", flush=True)
        elif "delta_energy_eV" in cmp:
            print(
                f"  vs reference: dE = {cmp['delta_energy_eV']:.6f} eV "
                f"(|dE| = {cmp['abs_delta_energy_eV']:.6f} eV)",
                flush=True,
            )
        elif cmp.get("mean_delta_energy_eV") is not None:
            print(
                f"  vs reference: mean dE = {cmp['mean_delta_energy_eV']:.6f} eV "
                f"(RMSE = {cmp['rmse_delta_energy_eV']:.6f} eV)",
                flush=True,
            )
        if "force_rmse_eV_A" in cmp:
            print(
                f"  vs reference: force RMSE = {cmp['force_rmse_eV_A']:.6f} eV/A",
                flush=True,
            )
        elif cmp.get("mean_force_rmse_eV_A") is not None:
            print(
                f"  vs reference: mean force RMSE = {cmp['mean_force_rmse_eV_A']:.6f} eV/A",
                flush=True,
            )


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


def _reference_z_for_frame(reference: Any, frame: int, n_atoms: int) -> np.ndarray:
    z_arr = np.asarray(reference.Z)
    if z_arr.ndim == 1:
        return np.asarray(z_arr[:n_atoms], dtype=np.int32)
    return np.asarray(z_arr[int(frame), :n_atoms], dtype=np.int32)


def center_positions_at_com(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    """Translate positions so the mass-weighted center of mass is at the origin."""
    from ase.data import atomic_masses

    pos = np.asarray(positions, dtype=np.float64)
    z = np.asarray(atomic_numbers, dtype=np.int32).reshape(-1)
    masses = np.asarray(atomic_masses[z], dtype=np.float64)
    weights = np.ones(len(z), dtype=np.float64) if float(masses.sum()) <= 0.0 else masses
    if pos.ndim == 2:
        return pos - np.average(pos, axis=0, weights=weights)
    com = np.average(pos, axis=1, weights=weights)
    return pos - com[:, np.newaxis, :]


def reference_metrics_at_eval_geometry(
    reference: Any,
    *,
    ref_frame: int,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
) -> tuple[float | None, np.ndarray | None]:
    """Reference energy (eV) and forces (eV/Å) aligned to the evaluate geometry."""
    n_atoms = int(len(atomic_numbers))
    ref_z = _reference_z_for_frame(reference, int(ref_frame), n_atoms)
    ref_r = np.asarray(reference.R[int(ref_frame), :n_atoms], dtype=np.float64)
    ref_f_raw = None
    if getattr(reference, "has_F", False) and getattr(reference, "F", None) is not None:
        ref_f_raw = np.asarray(reference.F[int(ref_frame), :n_atoms], dtype=np.float64)
    _, ref_f_aligned, _ = align_reference_frame_to_evaluate(
        ref_r,
        ref_z,
        positions,
        atomic_numbers,
        reference_forces=ref_f_raw,
    )
    ref_e_ev = None
    if getattr(reference, "has_E", False) and getattr(reference, "E", None) is not None:
        ref_e_ev = float(
            energy_to_ev(float(reference.E[int(ref_frame)]), reference.energy_unit)
        )
    ref_f_ev = None
    if ref_f_aligned is not None:
        ref_f_ev = np.asarray(
            forces_to_ev_angstrom(ref_f_aligned, reference.force_unit),
            dtype=np.float64,
        )
    return ref_e_ev, ref_f_ev


def reference_metrics_from_npz(
    reference_path: Path,
    *,
    ref_frame: int,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    reference_energy_unit: str | None = None,
    reference_force_unit: str | None = None,
) -> tuple[float | None, np.ndarray | None]:
    """Load reference frame metrics from an NPZ path (single-frame evaluate)."""
    ref_path = Path(reference_path).expanduser().resolve()
    if reference_energy_unit is None:
        reference_energy_unit = infer_reference_energy_unit(ref_path)
    if reference_force_unit is None:
        reference_force_unit = infer_reference_force_unit(ref_path)
    with np.load(ref_path, allow_pickle=True) as ref:
        if "N" in ref.files:
            ref_n = int(np.asarray(ref["N"], dtype=int).reshape(-1)[int(ref_frame)])
        else:
            ref_n = int(np.asarray(ref["R"], dtype=np.float64).shape[1])
        ref_z = np.asarray(ref["Z"][int(ref_frame), :ref_n], dtype=np.int32)
        ref_r = np.asarray(ref["R"][int(ref_frame), :ref_n], dtype=np.float64)
        ref_f_raw = (
            np.asarray(ref["F"][int(ref_frame), :ref_n], dtype=np.float64)
            if "F" in ref.files
            else None
        )
        _, ref_f_aligned, _ = align_reference_frame_to_evaluate(
            ref_r,
            ref_z,
            positions,
            atomic_numbers,
            reference_forces=ref_f_raw,
        )
        ref_e_ev = None
        if "E" in ref.files or "E_eV" in ref.files:
            ref_e_ev, _, _ = reference_energy_ev_at_frame(
                ref,
                int(ref_frame),
                path=ref_path,
                energy_unit=reference_energy_unit,
            )
        ref_f_ev = None
        if ref_f_aligned is not None:
            ref_f_ev = np.asarray(
                forces_to_ev_angstrom(ref_f_aligned, reference_force_unit),
                dtype=np.float64,
            )
    return ref_e_ev, ref_f_ev


def save_evaluate_compare_extxyz_trajectories(
    out_dir: Path,
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    model_energies_eV: np.ndarray,
    model_forces_eV_A: np.ndarray | None,
    reference_energies_eV: np.ndarray | None,
    reference_forces_eV_A: np.ndarray | None,
    prefix: str = "evaluate",
) -> dict[str, str]:
    """Write COM-centered extxyz trajectories for reference, model, and force difference."""
    pos_centered = center_positions_at_com(positions, atomic_numbers)
    artifacts: dict[str, str] = {}

    model_path = out_dir / f"{prefix}_model.extxyz"
    save_evaluate_extxyz_multi(
        model_path,
        atomic_numbers=atomic_numbers,
        positions=pos_centered,
        energies_eV=model_energies_eV,
        forces_eV_A=model_forces_eV_A,
    )
    artifacts["extxyz_model"] = str(model_path)
    if prefix == "evaluate":
        legacy_path = out_dir / "evaluate.extxyz"
        legacy_path.write_bytes(model_path.read_bytes())
        artifacts["extxyz"] = str(legacy_path)
    else:
        artifacts["extxyz"] = str(model_path)

    if reference_energies_eV is not None and reference_forces_eV_A is not None:
        gt_path = out_dir / f"{prefix}_ground_truth.extxyz"
        save_evaluate_extxyz_multi(
            gt_path,
            atomic_numbers=atomic_numbers,
            positions=pos_centered,
            energies_eV=reference_energies_eV,
            forces_eV_A=reference_forces_eV_A,
        )
        artifacts["extxyz_ground_truth"] = str(gt_path)

        if model_forces_eV_A is not None:
            diff_forces = np.asarray(model_forces_eV_A, dtype=np.float64) - np.asarray(
                reference_forces_eV_A, dtype=np.float64
            )
            diff_energies = np.asarray(model_energies_eV, dtype=np.float64) - np.asarray(
                reference_energies_eV, dtype=np.float64
            )
            diff_path = out_dir / f"{prefix}_difference.extxyz"
            save_evaluate_extxyz_multi(
                diff_path,
                atomic_numbers=atomic_numbers,
                positions=pos_centered,
                energies_eV=diff_energies,
                forces_eV_A=diff_forces,
            )
            artifacts["extxyz_difference"] = str(diff_path)

    return artifacts


def _permutation_within_monomer_blocks(
    reference_numbers: np.ndarray,
    evaluator_numbers: np.ndarray,
    atoms_per_list: list[int],
) -> np.ndarray:
    """Map evaluator rows to reference rows when only monomer atom order differs."""
    ref_z = np.asarray(reference_numbers, dtype=np.int32).reshape(-1)
    eval_z = np.asarray(evaluator_numbers, dtype=np.int32).reshape(-1)
    perm = np.arange(len(eval_z), dtype=int)
    offset = 0
    for n_per in atoms_per_list:
        n_per = int(n_per)
        block_ref = ref_z[offset : offset + n_per]
        block_eval = eval_z[offset : offset + n_per]
        if not np.array_equal(block_ref, block_eval):
            block_perm = np.empty(n_per, dtype=int)
            for element in sorted(set(int(z) for z in block_eval.tolist())):
                eval_idx = np.where(block_eval == element)[0]
                ref_idx = np.where(block_ref == element)[0]
                if len(eval_idx) != len(ref_idx):
                    raise ValueError(
                        f"Element {element} count mismatch in monomer block at offset {offset}"
                    )
                for e_local, r_local in zip(eval_idx, ref_idx):
                    block_perm[e_local] = r_local
            perm[offset : offset + n_per] = offset + block_perm
        offset += n_per
    return perm


def _permutation_ref_to_evaluator_z(
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    evaluator_numbers: np.ndarray,
    *,
    geometry_hint: np.ndarray | None = None,
    atoms_per_list: list[int] | None = None,
    use_geometry: bool = True,
) -> np.ndarray:
    ref_r = np.asarray(reference_positions, dtype=np.float64).reshape(-1, 3)
    ref_z = np.asarray(reference_numbers, dtype=np.int32).reshape(-1)
    eval_z = np.asarray(evaluator_numbers, dtype=np.int32).reshape(-1)
    n_atoms = int(len(eval_z))
    if np.array_equal(ref_z, eval_z):
        return np.arange(n_atoms, dtype=int)
    if atoms_per_list and int(sum(atoms_per_list)) == n_atoms:
        if not use_geometry or geometry_hint is None:
            return _permutation_within_monomer_blocks(ref_z, eval_z, atoms_per_list)
        hint = np.asarray(geometry_hint, dtype=np.float64).reshape(-1, 3)
        perm = np.arange(n_atoms, dtype=int)
        offset = 0
        for n_per in atoms_per_list:
            n_per = int(n_per)
            block_ref_r = ref_r[offset : offset + n_per]
            block_ref_z = ref_z[offset : offset + n_per]
            block_hint = hint[offset : offset + n_per]
            block_eval_z = eval_z[offset : offset + n_per]
            if not np.array_equal(block_ref_z, block_eval_z):
                block_perm = _assign_atoms_by_element(
                    block_ref_r,
                    block_ref_z,
                    block_hint,
                    block_eval_z,
                )
                perm[offset : offset + n_per] = offset + block_perm
            offset += n_per
        return perm
    hint = (
        np.asarray(geometry_hint, dtype=np.float64).reshape(-1, 3)
        if geometry_hint is not None
        else ref_r
    )
    return _assign_atoms_by_element(ref_r, ref_z, hint, eval_z)


def positions_for_evaluator_z(
    reference_positions: np.ndarray,
    reference_numbers: np.ndarray,
    evaluator_numbers: np.ndarray,
    *,
    geometry_hint: np.ndarray | None = None,
    atoms_per_list: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return coordinates permuted into evaluator ``Z`` row order."""
    ref_r = np.asarray(reference_positions, dtype=np.float64).reshape(-1, 3)
    ref_z = np.asarray(reference_numbers, dtype=np.int32).reshape(-1)
    eval_z = np.asarray(evaluator_numbers, dtype=np.int32).reshape(-1)
    perm = _permutation_ref_to_evaluator_z(
        ref_r,
        ref_z,
        eval_z,
        geometry_hint=geometry_hint,
        atoms_per_list=atoms_per_list,
    )
    reordered = ref_r[perm]
    meta = {
        "geometry_reordered": not np.array_equal(perm, np.arange(len(eval_z))),
        "geometry_permutation": perm.tolist(),
    }
    if geometry_hint is not None:
        hint = np.asarray(geometry_hint, dtype=np.float64).reshape(-1, 3)
        meta["position_rmsd_after_reorder_A"] = float(
            np.sqrt(np.mean((reordered - hint) ** 2))
        )
    return reordered, meta


def permute_handoff_array_to_evaluator_z(
    values: np.ndarray,
    *,
    handoff_positions: np.ndarray,
    handoff_numbers: np.ndarray,
    evaluator_numbers: np.ndarray,
    atoms_per_list: list[int] | None = None,
) -> np.ndarray:
    """Reorder per-atom handoff arrays (charges, LJ types, …) to evaluator ``Z`` order."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    perm = _permutation_ref_to_evaluator_z(
        np.asarray(handoff_positions, dtype=np.float64).reshape(-1, 3),
        np.asarray(handoff_numbers, dtype=np.int32).reshape(-1),
        np.asarray(evaluator_numbers, dtype=np.int32).reshape(-1),
        atoms_per_list=atoms_per_list,
        use_geometry=False,
    )
    return flat[perm]


def _psf_z_for_composition(
    composition: str,
    *,
    expected_atoms: int,
    atoms_per_list: list[int],
    residue_labels: list[str],
) -> np.ndarray:
    from mmml.cli.run.md_pbc_suite.ase import _build_cluster_psf_topology_only, _parse_composition

    return _build_cluster_psf_topology_only(
        _parse_composition(composition),
        expected_atoms=expected_atoms,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
    )


def _resolve_evaluator_z_and_handoff_layout(
    *,
    handoff: MdHandoffState,
    z: np.ndarray,
    composition: str | None,
    atoms_per_list: list[int],
    residue_labels: list[str],
    evaluate_reference_npz: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prefer PSF/reference ``Z`` when a reference NPZ drives evaluation."""
    warnings: list[str] = []
    handoff_z = np.asarray(z, dtype=np.int32).copy()
    if composition is None or evaluate_reference_npz is None:
        return handoff_z, handoff_z, warnings

    z_psf = _psf_z_for_composition(
        composition,
        expected_atoms=len(handoff_z),
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
    )
    if np.array_equal(handoff_z, z_psf):
        return z_psf, handoff_z, warnings

    if sorted(handoff_z.tolist()) != sorted(z_psf.tolist()):
        warnings.append(
            "handoff atomic_numbers differ from composition PSF topology; "
            "keeping handoff Z"
        )
        return handoff_z, handoff_z, warnings

    warnings.append(
        "handoff atomic_numbers use a different atom order than PSF/reference; "
        "using PSF order for MMML evaluation"
    )
    return z_psf, handoff_z, warnings


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
        "_mmml_units": np.array(
            json.dumps({"E": "ev", "F": "ev_angstrom", "R": "angstrom", "E_eV": "ev"})
        ),
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
        "_mmml_units": np.array(
            json.dumps({"E": "ev", "F": "ev_angstrom", "R": "angstrom", "E_eV": "ev"})
        ),
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
    reference_energy_unit: str | None = None,
    reference_force_unit: str | None = None,
) -> dict[str, Any]:
    """Compare MMML single-point results to a reference trajectory NPZ frame."""
    ref_path = Path(reference_path).expanduser().resolve()
    if reference_energy_unit is None:
        reference_energy_unit = infer_reference_energy_unit(ref_path)
    if reference_force_unit is None:
        reference_force_unit = infer_reference_force_unit(ref_path)
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
        if "E" in ref.files or "E_eV" in ref.files:
            ref_e_ev, unit_used, ref_e_raw = reference_energy_ev_at_frame(
                ref,
                ref_frame,
                path=ref_path,
                energy_unit=reference_energy_unit,
            )
            delta_e = float(energy_eV) - ref_e_ev
            out["reference_energy_raw"] = ref_e_raw
            out["reference_energy_unit"] = unit_used
            out["reference_energy_eV"] = ref_e_ev
            out["predicted_energy_eV"] = float(energy_eV)
            out["delta_energy_eV"] = delta_e
            out["abs_delta_energy_eV"] = abs(delta_e)
        if forces_eV_A is not None and ref_f_raw is not None:
            ref_f_ev = np.asarray(
                forces_to_ev_angstrom(ref_f_raw, reference_force_unit),
                dtype=np.float64,
            )
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
    raw = {
        "energy_eV": float(atoms.get_potential_energy()),
        "forces_eV_A": np.asarray(atoms.get_forces(), dtype=np.float64),
        "n_atoms": int(len(z)),
        "n_monomers": int(n_monomers),
        "pbc": bool(use_pbc),
        "box_A": float(L) if L is not None else None,
    }
    return normalize_metrics_to_ev(raw, atoms=atoms)


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
    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
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
    ase_result = normalize_metrics_to_ev(
        {
            "energy_eV": float(atoms.get_potential_energy()),
            "forces_eV_A": np.asarray(atoms.get_forces(), dtype=np.float64),
            "n_atoms": int(len(z)),
            "n_monomers": int(n_monomers),
            "pbc": bool(use_pbc),
            "box_A": float(L) if L is not None else None,
            "path": "ase_interface",
        },
        atoms=atoms,
    )

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
    ase_result["jaxmd_jit"] = normalize_metrics_to_ev(
        {
            "energy_eV": energy_jit,
            "forces_eV_A": forces_jit,
        }
    )
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
        charmm_total_forces_ev_angstrom,
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
    refresh_ctx = "" if bool(getattr(args, "quiet", False)) else "evaluate-npz"
    grms = refresh_mlpot_energy_and_grms(ctx, context=refresh_ctx)
    charmm_row = charmm_energy_row()
    total_kcal = float(charmm_row.get("ENER", charmm_row.get("ENERGY", 0.0)))
    forces_ev = charmm_total_forces_ev_angstrom()[: int(len(z))]
    return normalize_metrics_to_ev(
        {
            "energy_kcal_mol": total_kcal,
            "energy_eV": total_kcal / float(ev2kcalmol),
            "forces_eV_A": forces_ev,
            "force_unit": "ev_angstrom",
            "grms_kcal_mol_A": float(grms),
            "charmm_energy_terms_kcal_mol": charmm_row,
            "n_atoms": int(len(z)),
            "n_monomers": int(n_monomers),
            "pbc": bool(use_pbc),
            "box_A": float(L) if L is not None else None,
            "path": "pycharmm_mlpot_callback",
        }
    )


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
    ref_path = getattr(args, "evaluate_reference_npz", None)
    ref_path_resolved = (
        Path(ref_path).expanduser().resolve() if ref_path is not None else None
    )
    z, handoff_z, z_warnings = _resolve_evaluator_z_and_handoff_layout(
        handoff=handoff,
        z=z,
        composition=getattr(args, "composition", None),
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
        evaluate_reference_npz=ref_path_resolved,
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
    if ref_path_resolved is not None and should_evaluate_reference_trajectory(args):
        n_atoms_monomer = int(len(z) // n_monomers)
        reference = load_reference_trajectory_npz(
            ref_path_resolved,
            z_fallback=z,
            n_atoms_monomer=n_atoms_monomer,
            n_monomers=n_monomers,
            max_frames=resolve_evaluate_max_frames(args),
        )
        ref_frame0 = int(reference.frame_indices[0])
        ref_z0 = _reference_z_for_frame(reference, ref_frame0, len(z))
        ref_r0 = np.asarray(reference.R[ref_frame0, : len(z)], dtype=np.float64)
        r0, _ = positions_for_evaluator_z(
            ref_r0,
            ref_z0,
            z,
            geometry_hint=ref_r0,
            atoms_per_list=atoms_per_list,
        )
    elif ref_path_resolved is not None:
        with np.load(ref_path_resolved, allow_pickle=True) as ref_peek:
            if "R" in ref_peek.files:
                ref_frame = int(getattr(args, "evaluate_reference_frame", frame) or frame)
                if "Z" in ref_peek.files:
                    z_arr = np.asarray(ref_peek["Z"])
                    ref_z = (
                        np.asarray(z_arr[: len(z)], dtype=np.int32)
                        if z_arr.ndim == 1
                        else np.asarray(z_arr[ref_frame, : len(z)], dtype=np.int32)
                    )
                else:
                    ref_z = z
                ref_r = np.asarray(ref_peek["R"][ref_frame, : len(z)], dtype=np.float64)
                r0, _ = positions_for_evaluator_z(
                    ref_r,
                    ref_z,
                    z,
                    geometry_hint=ref_r,
                    atoms_per_list=atoms_per_list,
                )

    ensure_psf_for_handoff_cluster(
        composition=handoff_composition,
        atomic_numbers=z,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
        positions=r0,
        quiet=bool(getattr(args, "quiet", False)),
    )
    if payload.charges is not None:
        charges = payload.charges
        if not np.array_equal(handoff_z, z):
            charges = permute_handoff_array_to_evaluator_z(
                charges,
                handoff_positions=handoff.positions,
                handoff_numbers=handoff_z,
                evaluator_numbers=z,
                atoms_per_list=atoms_per_list,
            )
        apply_npz_charges_to_psf(charges)
    if payload.at_codes is not None and not np.array_equal(handoff_z, z):
        payload = EvaluateNpzPayload(
            handoff=payload.handoff,
            charges=payload.charges,
            at_codes=permute_handoff_array_to_evaluator_z(
                payload.at_codes,
                handoff_positions=handoff.positions,
                handoff_numbers=handoff_z,
                evaluator_numbers=z,
                atoms_per_list=atoms_per_list,
            ).astype(np.int32),
            epsilon=payload.epsilon,
            sigma=payload.sigma,
        )

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
        "handoff_z": handoff_z,
        "z_warnings": z_warnings,
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
    atoms_per_list = ctx["atoms_per_list"]
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
    ref_energies: list[float] = []
    ref_forces_list: list[np.ndarray] = []
    per_frame_compare: list[dict[str, Any]] = []
    ref_energy_unit, ref_force_unit = resolve_reference_units(reference.path, args)

    if not getattr(args, "quiet", False):
        print(
            f"mmml md-system evaluate-npz ({backend}): "
            f"evaluating {n_eval} frames from {reference.path.name}",
            flush=True,
        )
        print(
            f"  reference units: E={ref_energy_unit}, F={ref_force_unit}",
            flush=True,
        )
        for warning in ctx.get("z_warnings", []):
            print(f"  note: {warning}", flush=True)

    for ref_frame in frame_indices:
        ref_z = _reference_z_for_frame(reference, int(ref_frame), len(z))
        ref_r = np.asarray(reference.R[int(ref_frame), : len(z)], dtype=np.float64)
        pos, _geom_meta = positions_for_evaluator_z(
            ref_r,
            ref_z,
            z,
            geometry_hint=ref_r,
            atoms_per_list=atoms_per_list,
        )
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
        ref_e_ev, ref_f_ev = reference_metrics_at_eval_geometry(
            reference,
            ref_frame=int(ref_frame),
            atomic_numbers=z,
            positions=positions_out[-1],
        )
        if ref_e_ev is not None:
            ref_energies.append(ref_e_ev)
        if ref_f_ev is not None:
            ref_forces_list.append(ref_f_ev)
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
    ref_energies_arr = np.asarray(ref_energies, dtype=np.float64) if ref_energies else None
    ref_forces_arr = np.stack(ref_forces_list, axis=0) if ref_forces_list else None

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
        "reference_energy_unit": ref_energy_unit,
        "reference_force_unit": ref_force_unit,
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
        "units": dict(EVALUATE_ARTIFACT_UNITS),
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

        traj_prefix = Path(
            getattr(args, "evaluate_traj", None) or out_dir / "evaluate.extxyz"
        ).stem
        artifacts.update(
            save_evaluate_compare_extxyz_trajectories(
                out_dir,
                atomic_numbers=z,
                positions=positions_arr,
                model_energies_eV=energies_arr,
                model_forces_eV_A=forces_arr,
                reference_energies_eV=ref_energies_arr,
                reference_forces_eV_A=ref_forces_arr,
                prefix=traj_prefix,
            )
        )

        compare_path = getattr(args, "evaluate_compare_output", None) or out_dir / "evaluate_compare.json"
        compare_path = Path(compare_path).expanduser()
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")
        artifacts["compare_json"] = str(compare_path)

    if artifacts:
        result["artifacts"] = artifacts
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_evaluate_npz_summary(
        backend=backend,
        metrics=result.get("metrics", {}),
        out_path=out_path,
        artifacts=artifacts,
        result=result,
        quiet=bool(getattr(args, "quiet", False)),
        n_eval_frames=n_eval,
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
        "units": dict(EVALUATE_ARTIFACT_UNITS),
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

        traj_prefix = Path(
            getattr(args, "evaluate_traj", None) or out_dir / "evaluate.extxyz"
        ).stem
        ref_e_ev: float | None = None
        ref_f_ev: np.ndarray | None = None
        ref_path = getattr(args, "evaluate_reference_npz", None) or getattr(
            args, "reference_npz", None
        )
        if ref_path is not None:
            ref_frame = int(getattr(args, "evaluate_reference_frame", frame) or frame)
            ref_e_unit, ref_f_unit = resolve_reference_units(ref_path, args)
            ref_e_ev, ref_f_ev = reference_metrics_from_npz(
                Path(ref_path),
                ref_frame=ref_frame,
                atomic_numbers=z,
                positions=pos_out,
                reference_energy_unit=ref_e_unit,
                reference_force_unit=ref_f_unit,
            )
        artifacts.update(
            save_evaluate_compare_extxyz_trajectories(
                out_dir,
                atomic_numbers=z,
                positions=pos_out.reshape(1, -1, 3),
                model_energies_eV=np.array([energy_eV], dtype=np.float64),
                model_forces_eV_A=(
                    np.asarray(forces_eV_A, dtype=np.float64).reshape(1, -1, 3)
                    if forces_eV_A is not None
                    else None
                ),
                reference_energies_eV=(
                    np.array([ref_e_ev], dtype=np.float64) if ref_e_ev is not None else None
                ),
                reference_forces_eV_A=(
                    ref_f_ev.reshape(1, -1, 3) if ref_f_ev is not None else None
                ),
                prefix=traj_prefix,
            )
        )

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
        ref_e_unit, ref_f_unit = resolve_reference_units(ref_path, args)
        result["reference_units"] = {
            "energy": ref_e_unit,
            "force": ref_f_unit,
        }
        try:
            compare = compare_evaluate_to_reference_npz(
                Path(ref_path),
                frame=ref_frame,
                atomic_numbers=z,
                positions=pos_out,
                energy_eV=energy_eV,
                forces_eV_A=forces_eV_A,
                reference_energy_unit=ref_e_unit,
                reference_force_unit=ref_f_unit,
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
    _print_evaluate_npz_summary(
        backend=backend,
        metrics=metrics,
        out_path=out_path,
        artifacts=artifacts,
        result=result,
        quiet=bool(getattr(args, "quiet", False)),
    )
    return 0
