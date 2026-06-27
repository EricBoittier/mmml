#!/usr/bin/env python
"""
Compare reference (PySCF/QM) and model NPZ trajectories with metrics and plots.

Modes
-----
1. Two NPZ files (reference labels vs model predictions):
       mmml compare-npz --reference ref.npz --predictions pred.npz -o out/

2. Checkpoint inference against labeled NPZ (same file holds R,Z,E,F,...):
       mmml compare-npz --checkpoint params.json --data test.npz -o out/ --max-frames 200

Issue #12: per-atom / per-element force analysis and richer validation plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mmml.analysis.npz_comparison import (
    align_npz_arrays,
    compare_npz_arrays,
    plot_comparison,
    write_comparison_report,
)
from mmml.data.units import infer_reference_energy_unit


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def _run_physnet_checkpoint_inference(
    checkpoint: Path,
    data: dict[str, np.ndarray],
    *,
    max_frames: int | None,
    stride: int,
    natoms: int | None,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Batched PhysNet inference (matches ``mmml physnet-evaluate``)."""
    import jax

    from mmml.cli.misc.physnet_evaluate import (
        _ensure_npz_with_N,
        _infer_natoms_from_npz,
        _load_physnet_checkpoint,
    )
    from mmml.models.physnetjax.physnetjax.analysis.analysis import eval as physnet_eval
    from mmml.models.physnetjax.physnetjax.data.batches import prepare_batches_jit
    from mmml.models.physnetjax.physnetjax.data.data import prepare_datasets

    if "R" not in data or "Z" not in data or "E" not in data:
        raise KeyError("Data NPZ must contain R, Z, and E for checkpoint evaluation.")

    r = np.asarray(data["R"], dtype=np.float64)
    n_total = r.shape[0]
    indices = np.arange(0, n_total, int(stride), dtype=int)
    if max_frames is not None and len(indices) > int(max_frames):
        indices = indices[: int(max_frames)]

    # Subset NPZ for physnet data loaders
    subset = {k: (v[indices] if hasattr(v, "__len__") and len(v) == n_total else v) for k, v in data.items()}
    tmp_path, tmp_created = _ensure_npz_with_N(
        _write_temp_npz(subset)
    )
    try:
        nat = natoms if natoms is not None else _infer_natoms_from_npz(tmp_path)
        _, params, model = _load_physnet_checkpoint(checkpoint, nat)

        key = jax.random.PRNGKey(seed)
        _, valid_data = prepare_datasets(
            key,
            train_size=0,
            valid_size=len(indices),
            files=[str(tmp_path)],
            natoms=nat,
            verbose=False,
        )
        data_keys = ["R", "Z", "F", "E", "N"]
        if getattr(model, "charges", False) and "D" in valid_data:
            data_keys.append("D")

        key2, _ = jax.random.split(key)
        batches = prepare_batches_jit(
            key2,
            valid_data,
            batch_size,
            data_keys=data_keys,
            num_atoms=nat,
        )
        if not batches:
            raise ValueError(
                "No full batches (lower --batch-size or evaluate more frames)."
            )

        Es, _, predEs, Fs, predFs, Ds, predDs, _, _ = physnet_eval(
            batches, model, params, batch_size=batch_size
        )

        n_eval = len(batches) * batch_size
        natoms_model = int(model.natoms)
        f_ref = Fs.reshape(n_eval, natoms_model, 3)
        f_pred = predFs.reshape(n_eval, natoms_model, 3)

        reference = {
            "E": np.asarray(Es, dtype=np.float64),
            "F": f_ref,
            "Z": np.asarray(valid_data["Z"][:n_eval], dtype=np.int32),
            "R": np.asarray(valid_data["R"][:n_eval], dtype=np.float64),
        }
        predictions = {
            "E": np.asarray(predEs, dtype=np.float64),
            "F": f_pred,
            "Z": reference["Z"],
            "R": reference["R"],
        }
        if getattr(model, "charges", False) and Ds is not None:
            reference["D"] = np.asarray(Ds[:n_eval], dtype=np.float64)
            predictions["D"] = np.asarray(predDs[:n_eval], dtype=np.float64)

        return {"reference": reference, "predictions": predictions, "indices": indices[:n_eval]}
    finally:
        if tmp_created:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_temp_npz(data: dict[str, np.ndarray]) -> Path:
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    path = Path(tmp.name)
    tmp.close()
    np.savez(path, **data)
    return path


def _run_checkpoint_inference(
    checkpoint: Path,
    data: dict[str, np.ndarray],
    *,
    max_frames: int | None,
    stride: int,
    cutoff: float | None,
    use_dcmnet_dipole: bool,
) -> dict[str, np.ndarray]:
    from ase import Atoms

    from mmml.interfaces.calculators.checkpoint_loading import (
        create_calculator_from_checkpoint,
    )

    if "R" not in data or "Z" not in data or "E" not in data:
        raise KeyError("Data NPZ must contain R, Z, and E for checkpoint evaluation.")

    r = np.asarray(data["R"], dtype=np.float64)
    z = np.asarray(data["Z"])
    n_total = r.shape[0]
    indices = np.arange(0, n_total, int(stride), dtype=int)
    if max_frames is not None and len(indices) > int(max_frames):
        indices = indices[: int(max_frames)]

    calc = create_calculator_from_checkpoint(
        checkpoint,
        cutoff=cutoff,
        use_dcmnet_dipole=use_dcmnet_dipole,
    )

    # Padded PhysNet checkpoints expect a fixed width (natoms); use full NPZ rows.
    model_natoms = getattr(getattr(calc, "model", None), "natoms", None)

    energies: list[float] = []
    forces: list[np.ndarray] = []
    dipoles: list[np.ndarray] = []

    has_f_ref = "F" in data
    z_1d = z[0] if z.ndim == 2 else z

    for i, frame_idx in enumerate(indices):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  frame {i + 1}/{len(indices)} (index {frame_idx})")
        zi = z[frame_idx] if z.ndim == 2 else z
        ri = r[frame_idx]
        if model_natoms is not None and z.ndim == 2:
            n_pad = int(model_natoms)
            zi = np.asarray(zi[:n_pad], dtype=int)
            ri = np.asarray(ri[:n_pad], dtype=np.float64)
        elif z.ndim == 2:
            n_atoms = int(np.sum(zi > 0))
            zi = zi[:n_atoms]
            ri = ri[:n_atoms]
        else:
            n_atoms = len(zi)

        atoms = Atoms(numbers=zi, positions=ri)
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()))
        if has_f_ref:
            forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
        try:
            dipoles.append(np.asarray(atoms.calc.get_dipole_moment(), dtype=np.float64))
        except Exception:
            pass

    pred: dict[str, np.ndarray] = {
        "E": np.asarray(energies, dtype=np.float64),
        "Z": np.broadcast_to(z_1d, (len(indices), len(z_1d))).copy()
        if z.ndim == 1
        else z[indices],
        "R": r[indices],
    }
    if forces:
        if z.ndim == 1:
            pred["F"] = np.stack(forces, axis=0)
        else:
            n_pad = z.shape[1]
            f_pad = np.zeros((len(forces), n_pad, 3), dtype=np.float64)
            for j, f in enumerate(forces):
                f_pad[j, : f.shape[0]] = f
            pred["F"] = f_pad
    if dipoles and len(dipoles) == len(indices):
        pred["D"] = np.stack(dipoles, axis=0)

    ref = {k: data[k] for k in data}
    if z.ndim == 1:
        ref = {**ref, "Z": np.broadcast_to(z, (n_total, len(z)))}
    ref = {k: (v[indices] if hasattr(v, "__len__") and len(v) == n_total else v) for k, v in ref.items()}
    return {"reference": ref, "predictions": pred, "indices": indices}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare reference and model NPZ data (metrics + plots).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference NPZ (PySCF / QM labels)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Model prediction NPZ (E, F, D, ...)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Model checkpoint JSON/pkl/dir; run inference on --data",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Labeled NPZ for --checkpoint mode (R,Z,E,F,...)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("compare_npz_out"),
        help="Output directory for metrics and plots",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max structures to compare (default: all)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for --checkpoint mode (default: 1)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Model cutoff override for checkpoint inference",
    )
    parser.add_argument(
        "--use-dcmnet-dipole",
        action="store_true",
        help="Use DCMNet dipole from joint checkpoint",
    )
    parser.add_argument(
        "--energy-unit",
        type=str,
        default=None,
        help="Energy unit label for plots (default: infer from reference NPZ)",
    )
    parser.add_argument(
        "--force-unit",
        type=str,
        default="eV/Å",
        help="Force unit label for plots (default: eV/Å)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="With --checkpoint, save inference NPZ to output dir",
    )

    args = parser.parse_args()

    if args.checkpoint:
        if args.data is None:
            print("Error: --checkpoint requires --data", file=sys.stderr)
            return 1
        if not args.checkpoint.exists():
            print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 1
        if not args.data.exists():
            print(f"Error: data NPZ not found: {args.data}", file=sys.stderr)
            return 1

        print(f"Running checkpoint inference: {args.checkpoint}")
        data = _load_npz_dict(args.data)
        try:
            from mmml.cli.misc.physnet_evaluate import _resolve_physnet_json_path

            if _resolve_physnet_json_path(args.checkpoint) is not None:
                bundle = _run_physnet_checkpoint_inference(
                    args.checkpoint,
                    data,
                    max_frames=args.max_frames,
                    stride=args.stride,
                    natoms=None,
                    batch_size=4,
                    seed=0,
                )
            else:
                raise ValueError("not physnet json")
        except (ValueError, ImportError):
            bundle = _run_checkpoint_inference(
                args.checkpoint,
                data,
                max_frames=args.max_frames,
                stride=args.stride,
                cutoff=args.cutoff,
                use_dcmnet_dipole=args.use_dcmnet_dipole,
            )
        reference = bundle["reference"]
        predictions = bundle["predictions"]
        ref_path = str(args.data.resolve())
        pred_path = str(args.checkpoint.resolve())
        title_prefix = f"{args.checkpoint.name} vs labels | "
    elif args.reference and args.predictions:
        if not args.reference.exists():
            print(f"Error: reference not found: {args.reference}", file=sys.stderr)
            return 1
        if not args.predictions.exists():
            print(f"Error: predictions not found: {args.predictions}", file=sys.stderr)
            return 1
        reference = _load_npz_dict(args.reference)
        predictions = _load_npz_dict(args.predictions)
        ref_path = str(args.reference.resolve())
        pred_path = str(args.predictions.resolve())
        title_prefix = ""
    else:
        print(
            "Error: provide --reference and --predictions, or --checkpoint and --data",
            file=sys.stderr,
        )
        return 1

    aligned = align_npz_arrays(
        reference,
        predictions,
        max_frames=args.max_frames,
    )
    energy_unit = args.energy_unit
    if energy_unit is None:
        try:
            energy_unit = infer_reference_energy_unit(
                args.data if args.checkpoint else args.reference
            )
        except Exception:
            energy_unit = "eV"

    metrics = compare_npz_arrays(
        aligned,
        energy_unit_label=energy_unit,
        force_unit_label=args.force_unit,
    )

    out_dir = Path(args.output_dir)
    plot_paths = []
    if not args.no_plots:
        try:
            plot_paths = plot_comparison(
                aligned,
                metrics,
                out_dir,
                title_prefix=title_prefix,
                energy_unit=energy_unit,
                force_unit=args.force_unit,
            )
        except ImportError:
            print("Warning: matplotlib not available; skipping plots", file=sys.stderr)

    metrics_path = write_comparison_report(
        metrics,
        out_dir,
        reference=ref_path,
        predictions=pred_path,
        plot_paths=plot_paths,
    )

    if args.save_predictions and args.checkpoint:
        pred_npz = out_dir / "predictions.npz"
        np.savez_compressed(pred_npz, **predictions)
        print(f"Saved predictions to {pred_npz}")

    print(f"\nCompared {metrics['n_frames']} frames")
    e = metrics["energy"]
    print(f"  Energy  MAE={e['mae']:.6g}  RMSE={e['rmse']:.6g}  R²={e['r2']:.4f}")
    if "forces" in metrics:
        f = metrics["forces"]
        print(
            f"  Forces  MAE={f['mae']:.6g}  RMSE={f['rmse']:.6g}  R²={f['r2']:.4f}"
        )
        if "per_element_forces" in metrics:
            print("  Per-element force MAE (‖ΔF‖):")
            for sym, row in sorted(metrics["per_element_forces"].items()):
                print(f"    {sym}: n={row['n']}  MAE={row['mae']:.6g}")
    if "dipole" in metrics:
        d = metrics["dipole"]
        print(f"  Dipole  MAE={d['mae']:.6g}  R²={d['r2']:.4f}")

    print(f"\nWrote {metrics_path}")
    if plot_paths:
        print(f"Plots in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
