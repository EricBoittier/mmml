#!/usr/bin/env python3
"""Finite-difference force check for the PyCHARMM MLpot callback.

This script builds a small CHARMM cluster, registers the same decomposed MLpot
used by ``mmml md-system --backend pycharmm``, and compares callback forces
against central differences of the callback energy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from mmml.interfaces.pycharmmInterface.cutoffs import add_handoff_cutoff_args
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        add_cluster_args,
        add_charmm_output_args,
    )

    parser = argparse.ArgumentParser(
        description="Numerically test PyCHARMM MLpot callback forces."
    )
    parser.add_argument(
        "composition_arg",
        nargs="?",
        help="Shortcut for --composition, e.g. DCM:9 or ACO:4.",
    )
    add_cluster_args(parser)
    add_charmm_output_args(parser)
    add_handoff_cutoff_args(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pycharmm_mlpot/mlpot_force_fd.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pycharmm_mlpot/force_fd"),
        help="Scratch/output directory for cluster builders such as Packmol.",
    )
    parser.add_argument("--free-space", action="store_true", help="Disable PBC.")
    parser.add_argument("--box-size", type=float, default=None, help="Cubic PBC side in Å.")
    parser.add_argument(
        "--packmol-sphere",
        action="store_true",
        default=None,
        help="Use Packmol sphere placement for --composition.",
    )
    parser.add_argument("--packmol-radius", type=float, default=None, help="Packmol sphere radius in Å.")
    parser.add_argument("--packmol-tolerance", type=float, default=2.0, help="Packmol tolerance in Å.")
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        help="Radius hint used to infer Packmol sphere radius.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Packmol/random seed.")
    parser.add_argument("--charmm-sd-steps", type=int, default=50)
    parser.add_argument("--charmm-abnr-steps", type=int, default=100)
    parser.add_argument("--charmm-tolenr", type=float, default=1.0e-3)
    parser.add_argument("--charmm-tolgrd", type=float, default=1.0e-3)
    parser.add_argument("--ml-batch-size", type=int, default=None)
    parser.add_argument("--ml-gpu-count", type=int, default=None)
    parser.add_argument("--ml-max-active-dimers", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["ml-only", "monomer-only", "callback"],
        default="ml-only",
        help=(
            "What to finite-difference: 'ml-only' checks the decomposed MLpot "
            "JAX energy/forces with MM disabled; 'monomer-only' disables dimer "
            "terms; 'callback' checks the full PyCHARMM callback including "
            "stateful MM pair updates."
        ),
    )
    parser.add_argument(
        "--fd-step",
        type=float,
        default=1.0e-4,
        help="Central-difference displacement in Å.",
    )
    parser.add_argument(
        "--fd-steps",
        type=str,
        default="",
        help=(
            "Comma/space-separated FD displacements in Å. If set, run a step "
            "sweep and ignore --fd-step."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=5.0e-3,
        help="Absolute force tolerance in kcal/mol/Å.",
    )
    parser.add_argument(
        "--resids",
        type=str,
        default="",
        help="Comma/space-separated residue ids to test. Default: all atoms.",
    )
    parser.add_argument(
        "--fd-atoms",
        type=int,
        default=None,
        help="Limit to the first N atoms in the selected set.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Limit the number of xyz components checked after atom selection.",
    )
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_true",
        help="Always exit 0 after writing the report.",
    )
    args = parser.parse_args(argv)

    if args.composition_arg:
        if args.composition is not None and args.composition != args.composition_arg:
            parser.error("pass composition either positionally or with --composition, not both")
        args.composition = args.composition_arg
    return args


def _setup_charmm(args: argparse.Namespace, positions: np.ndarray) -> float | None:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_charmm_output_from_args,
        resolve_pbc_box_side,
        resolve_use_pbc,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    apply_charmm_output_from_args(args)
    if not resolve_use_pbc(args):
        setup_default_nbonds()
        return None
    box_side = resolve_pbc_box_side(args, positions)
    setup_charmm_environment(use_pbc=True, cubic_box_side_A=box_side)
    return float(box_side)


def _register_model(
    args: argparse.Namespace,
    z: np.ndarray,
    positions: np.ndarray,
    n_monomers: int,
    box_side: float | None,
) -> tuple[Any, Any]:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_checkpoint
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import _register_mlpot_context

    ckpt = resolve_checkpoint(args.checkpoint)
    return _register_mlpot_context(
        z,
        positions,
        ckpt,
        len(z),
        n_monomers,
        ml_batch_size=args.ml_batch_size,
        ml_gpu_count=args.ml_gpu_count,
        ml_max_active_dimers=args.ml_max_active_dimers,
        cubic_box_side_A=box_side,
        verbose=not args.quiet,
        args=args,
    )


def _empty_int() -> np.ndarray:
    return np.asarray([], dtype=np.int32)


def _callback_energy_and_force(calc: Any, positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(energy_kcal_mol, force_kcal_mol_A)`` from MLpot callback arrays."""
    n = int(len(positions))
    pos = np.asarray(positions, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dz = np.zeros(n, dtype=np.float64)
    energy = calc.calculate_charmm(
        Natom=n,
        Ntrans=0,
        Natim=n,
        idxp=np.arange(n, dtype=np.int32),
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        dx=dx,
        dy=dy,
        dz=dz,
        Nmlp=0,
        Nmlmmp=0,
        idxi=_empty_int(),
        idxj=_empty_int(),
        idxjp=_empty_int(),
        idxu=_empty_int(),
        idxv=_empty_int(),
        idxup=_empty_int(),
        idxvp=_empty_int(),
    )
    gradient = np.column_stack([dx, dy, dz])
    return float(energy), -gradient


def _ml_energy_and_force(
    calc: Any,
    positions: np.ndarray,
    *,
    do_ml_dimer: bool,
) -> tuple[float, np.ndarray]:
    """Return ML-only decomposed energy/forces from the scalar energy gradient."""
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.jax_device_policy import mlpot_jax_device_context
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import cubic_box_matrix_from_side

    pos = np.asarray(positions, dtype=np.float64)
    n = int(len(pos))
    box = None
    if getattr(calc, "_cell", False):
        box = jnp.asarray(cubic_box_matrix_from_side(float(calc._cell)))
    with mlpot_jax_device_context():
        positions_jax = jnp.asarray(pos)
        atomic_numbers_jax = jnp.asarray(calc.atomic_numbers[:n])

        def energy_scalar(p):
            out = calc.spherical_fn(
                positions=p,
                atomic_numbers=atomic_numbers_jax,
                n_monomers=calc.n_monomers,
                cutoff_params=calc.cutoff_params,
                doML=True,
                doMM=False,
                doML_dimer=do_ml_dimer,
                box=box,
            )
            return jnp.reshape(out.energy, (-1,))[0]

        energy_raw, grad = jax.value_and_grad(energy_scalar)(positions_jax)
        energy = float(jax.device_get(energy_raw)) * float(calc.ev2kcal)
        forces = -np.asarray(jax.device_get(grad), dtype=np.float64) * float(calc.ev2kcal)
    return energy, forces


def _ml_only_energy_and_force(calc: Any, positions: np.ndarray) -> tuple[float, np.ndarray]:
    return _ml_energy_and_force(calc, positions, do_ml_dimer=True)


def _monomer_only_energy_and_force(calc: Any, positions: np.ndarray) -> tuple[float, np.ndarray]:
    return _ml_energy_and_force(calc, positions, do_ml_dimer=False)


def _selected_atom_indices(args: argparse.Namespace, n_atoms: int) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import parse_resid_list
    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_resids

    resids = tuple(parse_resid_list(args.resids))
    if resids:
        raw = np.asarray(select_by_resids(resids).get_atom_indexes(), dtype=int)
        atoms = raw - 1
        atoms = atoms[(atoms >= 0) & (atoms < n_atoms)]
        if atoms.size == 0:
            raise ValueError(f"--resids {args.resids!r} selected no atoms")
    else:
        atoms = np.arange(n_atoms, dtype=int)
    if args.fd_atoms is not None:
        atoms = atoms[: max(0, int(args.fd_atoms))]
    return atoms


def _force_fd_check(
    calc: Any,
    positions: np.ndarray,
    atom_indices: np.ndarray,
    *,
    mode: str,
    step: float,
    tol: float,
    max_components: int | None,
) -> dict[str, Any]:
    pos0 = np.asarray(positions, dtype=np.float64)
    if mode == "callback":
        evaluate = _callback_energy_and_force
    elif mode == "ml-only":
        evaluate = _ml_only_energy_and_force
    elif mode == "monomer-only":
        evaluate = _monomer_only_energy_and_force
    else:
        raise ValueError(f"unknown force-check mode: {mode!r}")
    e0, f_analytic = evaluate(calc, pos0)

    components = [(int(i), axis) for i in atom_indices for axis in range(3)]
    if max_components is not None:
        components = components[: max(0, int(max_components))]
    if not components:
        raise ValueError("no force components selected")

    rows: list[dict[str, float | int | str]] = []
    diffs = []
    worst: dict[str, float | int | str] | None = None
    for atom, axis in components:
        plus = pos0.copy()
        minus = pos0.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        ep, _ = evaluate(calc, plus)
        em, _ = evaluate(calc, minus)
        f_fd = -(ep - em) / (2.0 * step)
        f_an = float(f_analytic[atom, axis])
        diff = float(f_fd - f_an)
        abs_diff = abs(diff)
        row = {
            "atom": atom + 1,
            "axis": "xyz"[axis],
            "analytic_force_kcalmol_A": f_an,
            "fd_force_kcalmol_A": float(f_fd),
            "energy_plus_kcalmol": float(ep),
            "energy_minus_kcalmol": float(em),
            "energy_delta_kcalmol": float(ep - em),
            "diff_kcalmol_A": diff,
            "abs_diff_kcalmol_A": abs_diff,
        }
        rows.append(row)
        diffs.append(abs_diff)
        if np.isfinite(abs_diff) and (
            worst is None or abs_diff > float(worst["abs_diff_kcalmol_A"])
        ):
            worst = row

    diff_arr = np.asarray(diffs, dtype=float)
    finite_diff_arr = diff_arr[np.isfinite(diff_arr)]
    n_fail = int(np.count_nonzero(diff_arr > float(tol)))
    n_nonfinite = int(diff_arr.size - finite_diff_arr.size)
    max_abs_diff = float(finite_diff_arr.max()) if finite_diff_arr.size else float("nan")
    rms_abs_diff = (
        float(np.sqrt(np.mean(finite_diff_arr**2)))
        if finite_diff_arr.size
        else float("nan")
    )
    return {
        "mode": mode,
        "energy_kcalmol": e0,
        "fd_step_A": float(step),
        "tol_kcalmol_A": float(tol),
        "n_atoms_total": int(len(pos0)),
        "n_atoms_selected": int(len(atom_indices)),
        "n_components_checked": int(len(components)),
        "n_fail": n_fail,
        "n_nonfinite": n_nonfinite,
        "max_abs_diff_kcalmol_A": max_abs_diff,
        "rms_abs_diff_kcalmol_A": rms_abs_diff,
        "worst": worst,
        "components": rows,
    }


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).replace(",", " ").split():
        value = float(part)
        if value <= 0:
            raise ValueError(f"FD steps must be > 0, got {value}")
        values.append(value)
    return values


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        build_cluster_from_args_with_tag,
        print_cluster_geometry_summary,
    )

    z, positions, n_monomers, tag = build_cluster_from_args_with_tag(args)
    if not args.quiet:
        print_cluster_geometry_summary(positions, n_monomers)
    box_side = _setup_charmm(args, positions)
    ctx, pyc_model = _register_model(args, z, positions, n_monomers, box_side)
    try:
        calc = pyc_model.get_pycharmm_calculator()
        atom_indices = _selected_atom_indices(args, len(z))
        steps = _parse_float_list(args.fd_steps) if args.fd_steps else [float(args.fd_step)]
        reports = [
            _force_fd_check(
                calc,
                positions,
                atom_indices,
                mode=args.mode,
                step=step,
                tol=float(args.tol),
                max_components=args.max_components,
            )
            for step in steps
        ]
        report = reports[0] if len(reports) == 1 else {"sweep": reports}
        report["tag"] = tag
        report["composition"] = args.composition
        report["box_side_A"] = box_side
        if len(reports) == 1:
            report["passed"] = int(report["n_fail"]) == 0 and int(report["n_nonfinite"]) == 0
        else:
            report["passed"] = all(
                int(r["n_fail"]) == 0 and int(r["n_nonfinite"]) == 0 for r in reports
            )
    finally:
        ctx.unset()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    display_report = {k: v for k, v in report.items() if k != "components"}
    if "sweep" in display_report:
        display_report["sweep"] = [
            {k: v for k, v in r.items() if k != "components"} for r in report["sweep"]
        ]
    worst = report["worst"] if "worst" in report else None
    print(json.dumps(display_report, indent=2))
    if worst is not None:
        print(
            "Worst component: atom {atom} {axis}, analytic={analytic_force_kcalmol_A:.6f}, "
            "fd={fd_force_kcalmol_A:.6f}, abs_diff={abs_diff_kcalmol_A:.6f} kcal/mol/Å".format(
                **worst
            )
        )
    print(f"Wrote {args.output}")

    if report["passed"] or args.no_fail_on_mismatch:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
