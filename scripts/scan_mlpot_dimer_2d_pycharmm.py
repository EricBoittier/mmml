#!/usr/bin/env python3
"""PyCHARMM MLpot 2D dimer/trimer scan with component energies.

Builds an N-monomer cluster via the same CLI/cluster path as
``mmml md-system --backend pycharmm``, registers ``MLpot``, and records
energies on a 2D grid of COM separations. For dimers only d01 changes
geometry; d02 is retained as a grid axis for a consistent output shape.

For each grid point the script saves:
  * decomposed ML/MM hybrid terms (same model as MLpot USER)
  * CHARMM ``ENER`` / ``USER`` / ``VDW`` / ``ELEC`` after ``ENER``
  * callback energy from ``calculate_charmm``

Examples
--------
  export MMML_CKPT=/path/to/dcm_ckpt
  ./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \\
    DCM:2 --scan-1d --scan-tag pbc_jax_pme_ewald \\
    --box-size 36 --mlpot-pbc --lr-solver jax_pme --jax-pme-method ewald \\
    --output-dir artifacts/dimer_lr_scans

  # Full DCM + ACO solver sweep:
  ./scripts/run_dcm_aco_dimer_lr_scans.sh

  ./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \\
    DCM:3 --output-dir artifacts/pycharmm_mlpot/dimer_2d_scan/dcm3

  ./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \\
    --composition ACO:3 --checkpoint /path/to/aco_ckpt \\
    --scan-2d-min 3.0 --scan-2d-max 10.0 --scan-2d-steps 11

  # Multiple compositions (one NPZ each):
  ./scripts/mmml-charmm-mpirun.sh python scripts/scan_mlpot_dimer_2d_pycharmm.py \\
    --batch-compositions DCM:2,ACO:2 --checkpoint "$MMML_CKPT"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

EV_PER_KCAL = 1.0 / 23.0605


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from mmml.interfaces.pycharmmInterface.cutoffs import add_handoff_cutoff_args
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        add_charmm_output_args,
        add_cluster_args,
        add_mlpot_lr_nonbond_args,
        add_packmol_cache_args,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "composition_arg",
        nargs="?",
        help="Shortcut for --composition, e.g. DCM:2 or DCM:3.",
    )
    add_cluster_args(p)
    add_charmm_output_args(p)
    add_handoff_cutoff_args(p)
    add_packmol_cache_args(p)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "artifacts" / "pycharmm_mlpot" / "dimer_2d_scan",
        help="Output directory (per-run subdir when batching)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit NPZ path (overrides --output-dir layout)",
    )
    p.add_argument(
        "--batch-compositions",
        type=str,
        default="",
        help="Comma-separated RES:N entries to scan in one invocation (each needs N>=2)",
    )
    p.add_argument(
        "--packmol-sphere",
        action="store_true",
        default=None,
        help="Use Packmol sphere placement for --composition.",
    )
    p.add_argument("--packmol-radius", type=float, default=None)
    p.add_argument("--packmol-tolerance", type=float, default=2.0)
    p.add_argument("--flat-bottom-radius", type=float, default=None)
    p.add_argument(
        "--min-com-restraint-distance",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Enable a pairwise inter-monomer COM lower-wall restraint below "
            "this distance in Å."
        ),
    )
    p.add_argument(
        "--min-com-restraint-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for --min-com-restraint-distance.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--free-space", action="store_true")
    p.add_argument("--box-size", type=float, default=None)
    p.add_argument("--charmm-sd-steps", type=int, default=50)
    p.add_argument("--charmm-abnr-steps", type=int, default=100)
    p.add_argument("--ml-batch-size", type=int, default=None)
    p.add_argument("--ml-gpu-count", type=int, default=None)
    p.add_argument("--ml-max-active-dimers", type=int, default=None)
    p.add_argument("--scan-2d-min", type=float, default=3.0)
    p.add_argument("--scan-2d-max", type=float, default=10.0)
    p.add_argument("--scan-2d-steps", type=int, default=15)
    p.add_argument(
        "--angle-02-deg",
        type=float,
        default=60.0,
        help="Angle (deg) of monomer-2 COM from +x in the 2D scan",
    )
    p.add_argument(
        "--scan-tag",
        type=str,
        default="",
        help="Output subfolder tag (e.g. lr_jax_pme_ewald) when sweeping solvers",
    )
    p.add_argument(
        "--scan-1d",
        action="store_true",
        help="For dimers (N=2): scan COM distance along d01 only (faster 1D sweep)",
    )
    p.add_argument("--mlpot-pbc", action="store_true", default=False)
    add_mlpot_lr_nonbond_args(p)
    p.add_argument("--no-mm", action="store_true", help="Skip MM terms in decomposed eval")
    # p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.composition_arg:
        if args.composition is not None and args.composition != args.composition_arg:
            p.error("pass composition either positionally or with --composition, not both")
        args.composition = args.composition_arg
    return args


def _parse_batch_compositions(text: str) -> list[str]:
    out: list[str] = []
    for part in str(text).replace(";", ",").split(","):
        tok = part.strip()
        if tok:
            out.append(tok)
    return out


def _composition_monomer_count(composition: str) -> int:
    from mmml.cli.run.md_pbc_suite.ase import _parse_composition

    return sum(count for _, count in _parse_composition(composition))


def _resolve_output_path(
    args: argparse.Namespace,
    *,
    tag: str,
    ckpt: Path,
    batch: bool,
) -> Path:
    if args.output is not None and not batch:
        return Path(args.output).expanduser().resolve()
    ckpt_tag = ckpt.name.replace(" ", "_")
    out_dir = Path(args.output_dir).expanduser().resolve() / ckpt_tag / tag
    scan_tag = str(getattr(args, "scan_tag", "") or "").strip()
    if scan_tag:
        out_dir = out_dir / scan_tag.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "scan_1d.npz" if getattr(args, "scan_1d", False) else "scan_2d.npz"
    return out_dir / fname


def _empty_int() -> np.ndarray:
    return np.asarray([], dtype=np.int32)


def _callback_energy_kcal(calc: Any, positions: np.ndarray) -> tuple[float, float]:
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
    fmax = float(np.linalg.norm(-gradient, axis=1).max())
    return float(energy), fmax


def _eval_decomposed(
    pyCModel: Any,
    cutoff: Any,
    positions: np.ndarray,
    z: np.ndarray,
    n_monomers: int,
    *,
    do_mm: bool,
) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.calculator_utils import ModelOutput

    calc = pyCModel.get_pycharmm_calculator()
    pos_j = jnp.asarray(positions, dtype=jnp.float64)
    z_j = jnp.asarray(z, dtype=int)

    def _one(do_ml: bool, do_ml_dimer: bool, do_mm_flag: bool) -> ModelOutput:
        return calc.spherical_fn(
            pos_j,
            z_j,
            n_monomers,
            cutoff,
            doML=do_ml,
            doMM=do_mm_flag,
            doML_dimer=do_ml_dimer,
            debug=False,
        )

    full = _one(True, True, do_mm)
    ml_internal = _one(True, False, False)
    ml_dimer_only = _one(True, True, False)

    def _scalar(x) -> float:
        v = jax.device_get(x)
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)

    e_fields = {
        "energy": _scalar(full.energy),
        "hybrid_energy": _scalar(full.hybrid_energy),
        "internal_E": _scalar(full.internal_E),
        "ml_2b_E": _scalar(full.ml_2b_E),
        "dH": _scalar(full.dH),
        "mm_E": _scalar(full.mm_E) if do_mm else 0.0,
        "flat_bottom_E": _scalar(full.flat_bottom_E),
        "com_restraint_E": _scalar(full.com_restraint_E),
        "ml_internal_only": _scalar(ml_internal.internal_E),
        "ml_2b_contrib": _scalar(ml_dimer_only.ml_2b_E),
    }
    rec = {f"{k}_kcal": float(v) * EV_PER_KCAL for k, v in e_fields.items()}
    rec["com_restraint_min_dist_A"] = _scalar(full.com_restraint_min_dist)
    return rec


def _make_evaluator(
    pyCModel: Any,
    cutoff: Any,
    z: np.ndarray,
    n_monomers: int,
    *,
    do_mm: bool,
) -> Any:
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_energy_row
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    import pycharmm

    calc = pyCModel.get_pycharmm_calculator()

    def eval_at(positions: np.ndarray) -> dict[str, float]:
        sync_charmm_positions(positions)
        pycharmm.lingo.charmm_script("ENER")
        terms = charmm_energy_row()
        e_cb, fmax_cb = _callback_energy_kcal(calc, positions)
        rec = _eval_decomposed(
            pyCModel, cutoff, positions, z, n_monomers, do_mm=do_mm
        )
        rec["charmm_ENER_kcal"] = float(terms.get("ENER", np.nan))
        rec["charmm_USER_kcal"] = float(terms.get("USER", np.nan))
        rec["charmm_VDW_kcal"] = float(terms.get("VDW", np.nan))
        rec["charmm_ELEC_kcal"] = float(terms.get("ELEC", np.nan))
        rec["callback_energy_kcal"] = e_cb
        rec["callback_force_max_kcal_A"] = fmax_cb
        rec["charmm_grms_kcal_A"] = float(terms.get("GRMS", np.nan))
        return rec

    return eval_at


def _run_one_scan(args: argparse.Namespace, composition: str) -> Path:
    from mmml.interfaces.pycharmmInterface.cutoffs import cutoff_parameters_from_args
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        apply_charmm_output_from_args,
        build_cluster_from_args_with_tag,
        print_cluster_geometry_summary,
        resolve_checkpoint,
        resolve_mlpot_use_pbc,
        resolve_pbc_box_side,
        resolve_use_pbc,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import setup_charmm_environment
    from mmml.interfaces.pycharmmInterface.mlpot.run_workflow import _register_mlpot_context
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds
    from mmml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
        atoms_per_monomer_from_psf,
        default_scan_2d_metric_keys,
        distance_report,
        run_scan_2d,
    )

    if _composition_monomer_count(composition) < 2:
        raise SystemExit(
            f"2D dimer scan requires at least 2 monomers; got {composition!r}. "
            "Use e.g. DCM:2, DCM:3, or ACO:3."
        )

    args.composition = composition
    ckpt = resolve_checkpoint(args.checkpoint)
    apply_charmm_output_from_args(args)

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401

    z, ref_pos, n_mol, tag = build_cluster_from_args_with_tag(args)
    if n_mol < 2:
        raise SystemExit(f"Expected at least 2 monomers after build, got {n_mol}")

    atoms_per = atoms_per_monomer_from_psf()
    if len(atoms_per) != n_mol or sum(atoms_per) != len(z):
        raise SystemExit(f"PSF monomer layout mismatch: atoms_per={atoms_per}, natoms={len(z)}")

    if not args.quiet:
        print(f"Checkpoint: {ckpt}", flush=True)
        print(f"Composition: {composition} tag={tag}", flush=True)
        print_cluster_geometry_summary(ref_pos, n_mol)

    if resolve_use_pbc(args):
        box_side = resolve_pbc_box_side(args, ref_pos)
        setup_charmm_environment(use_pbc=True, cubic_box_side_A=box_side)
    else:
        setup_default_nbonds()
        box_side = None

    mlpot_pbc = resolve_mlpot_use_pbc(args)
    cutoff = cutoff_parameters_from_args(args)
    ctx, pyCModel = _register_mlpot_context(
        z,
        ref_pos,
        ckpt,
        len(z),
        n_mol,
        atoms_per_monomer=atoms_per,
        ml_batch_size=args.ml_batch_size,
        ml_gpu_count=args.ml_gpu_count,
        ml_max_active_dimers=args.ml_max_active_dimers,
        cubic_box_side_A=box_side,
        mlpot_use_pbc=mlpot_pbc,
        verbose=not args.quiet,
        args=args,
    )

    try:
        from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import warmup_decomposed_mlpot

        warmup_decomposed_mlpot(
            pyCModel,
            ref_pos,
            cell=float(box_side) if mlpot_pbc and box_side is not None else None,
            verbose=not args.quiet,
        )

        ref_dist = distance_report(ref_pos, atoms_per)
        if not args.quiet:
            print(
                "Reference COM (Å): "
                + " ".join(
                    f"{key[4:]}={value:.3f}"
                    for key, value in sorted(ref_dist.items())
                    if key.startswith("com_")
                ),
                flush=True,
            )

        d1 = np.linspace(float(args.scan_2d_min), float(args.scan_2d_max), int(args.scan_2d_steps))
        if getattr(args, "scan_1d", False) and n_mol == 2:
            d2 = d1[:1]
        else:
            d2 = np.linspace(float(args.scan_2d_min), float(args.scan_2d_max), int(args.scan_2d_steps))
        metric_keys = default_scan_2d_metric_keys(include_mm=not args.no_mm)
        eval_fn = _make_evaluator(
            pyCModel, cutoff, z, n_mol, do_mm=not args.no_mm
        )

        def _progress(done: int, total: int) -> None:
            if not args.quiet and (done == 1 or done == total):
                print(f"  2D scan row {done}/{total} done", flush=True)

        scan_2d = run_scan_2d(
            eval_fn,
            ref_pos,
            atoms_per,
            d1,
            d2,
            angle_02_deg=float(args.angle_02_deg),
            metric_keys=metric_keys,
            progress=_progress,
        )
    finally:
        ctx.unset()

    out_path = _resolve_output_path(args, tag=tag, ckpt=ckpt, batch=bool(args.batch_compositions))
    from mmml.interfaces.pycharmmInterface.long_range_backend import pick_lr_solver

    lr_requested = getattr(args, "lr_solver", None)
    lr_active = pick_lr_solver(lr_requested)
    save: dict[str, Any] = {
        "composition": np.array(composition),
        "checkpoint": np.array(str(ckpt)),
        "backend": np.array("pycharmm"),
        "scan_tag": np.array(str(getattr(args, "scan_tag", "") or "")),
        "mm_nonbond_mode": np.array(str(getattr(args, "mm_nonbond_mode", "jax_mic") or "jax_mic")),
        "lr_solver_requested": np.array("" if lr_requested is None else str(lr_requested)),
        "lr_solver_active": np.array(str(lr_active)),
        "jax_pme_method": np.array(
            str(getattr(args, "jax_pme_method", None) or os.environ.get("JAX_PME_METHOD", "ewald"))
        ),
        "jax_pme_sr_cutoff": np.float64(getattr(args, "jax_pme_sr_cutoff", None) or 6.0),
        "mlpot_pbc": np.bool_(bool(getattr(args, "mlpot_pbc", False))),
        "box_size_A": np.float64(
            np.nan if getattr(args, "box_size", None) is None else float(args.box_size)
        ),
        "scan_1d": np.bool_(bool(getattr(args, "scan_1d", False))),
        "atoms_per_monomer": np.array(atoms_per, dtype=np.int32),
        "reference_positions_A": ref_pos.astype(np.float64),
        "atomic_numbers": z.astype(np.int32),
        "mm_switch_on": np.float64(getattr(args, "mm_switch_on", np.nan)),
        "mm_switch_width": np.float64(getattr(args, "mm_switch_width", np.nan)),
        "ml_switch_width": np.float64(getattr(args, "ml_switch_width", np.nan)),
        "min_com_restraint_distance": np.float64(
            np.nan
            if getattr(args, "min_com_restraint_distance", None) is None
            else args.min_com_restraint_distance
        ),
        "min_com_restraint_k": np.float64(getattr(args, "min_com_restraint_k", np.nan)),
        "do_mm": np.bool_(not args.no_mm),
        "scan_2d_d01_A": d1.astype(np.float64),
        "scan_2d_d02_A": d2.astype(np.float64),
        "angle_02_deg": np.float64(args.angle_02_deg),
    }
    for k, v in ref_dist.items():
        save[f"reference_{k}"] = np.float64(v)
    for k, v in scan_2d.items():
        save[f"scan_2d_{k}"] = v

    np.savez_compressed(out_path, **save)
    print(f"Wrote {out_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    compositions = _parse_batch_compositions(args.batch_compositions)
    if compositions:
        if args.composition and args.composition not in compositions:
            compositions = [args.composition] + compositions
        elif args.composition:
            compositions = [args.composition]
    elif args.composition:
        compositions = [args.composition]
    else:
        raise SystemExit("Provide --composition RES:N (N>=2) or --batch-compositions")

    for comp in compositions:
        _run_one_scan(args, comp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
