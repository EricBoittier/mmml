#!/usr/bin/env python3
"""
Evaluate a trained PhysNet (PhysNetJAX) checkpoint on an NPZ dataset.

Runs real model inference (orbax checkpoint + EF forward), reports energy / force /
dipole errors in kcal/mol (and eV where noted), optional parity plots.

Usage:
    mmml physnet-evaluate --checkpoint out/ckpts/run --data splits/test.npz -o eval_out/
    mmml physnet-evaluate --checkpoint out/ckpts/run --data splits/test.npz \\
        --natoms 64 --batch-size 32 --plots --num-samples 500
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _infer_natoms_from_npz(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    if "Z" in data.files:
        z = np.asarray(data["Z"])
        if z.ndim >= 2:
            return int(z.shape[1])
    if "R" in data.files:
        r = np.asarray(data["R"])
        if r.ndim >= 2:
            return int(r.shape[1])
    raise ValueError(
        f"Could not infer --natoms from {path} (need Z or R with shape [n, natoms, ...])."
    )


def _ensure_npz_with_N(path: Path) -> Tuple[Path, bool]:
    """
    Return a path to an NPZ that includes 'N' (atom counts per structure).

    If ``path`` already has N, returns (path, False). Otherwise builds a
    temporary NPZ with N = count(Z > 0, axis=1) and returns (temp_path, True).
    """
    raw = np.load(path, allow_pickle=True)
    if "N" in raw.files:
        return path, False
    if "Z" not in raw.files:
        raise ValueError(f"{path} has no 'N' and no 'Z'; cannot infer atom counts.")
    z = np.asarray(raw["Z"])
    n_per = np.sum(z > 0, axis=1).astype(np.int32)
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    payload = {k: np.asarray(raw[k]) for k in raw.files}
    payload["N"] = n_per
    np.savez(tmp_path, **payload)
    return tmp_path, True


def _count_samples(path: Path) -> int:
    data = np.load(path, allow_pickle=True)
    if "R" not in data.files:
        raise ValueError(f"{path} has no 'R' array.")
    r = np.asarray(data["R"])
    if r.ndim == 2:
        return int(r.shape[0])
    if r.ndim == 3:
        return int(r.shape[0])
    raise ValueError(f"Unexpected R shape {r.shape} in {path}")


def _normalize_valid_data(valid_data: Dict[str, Any]) -> None:
    """Align dipole key with PhysNet eval / batch code (expects 'D')."""
    if "D" not in valid_data:
        if "Dxyz" in valid_data:
            valid_data["D"] = valid_data["Dxyz"]
        elif "dipole" in valid_data:
            d = np.asarray(valid_data["dipole"])
            if d.ndim == 2 and d.shape[-1] == 3:
                valid_data["D"] = d


def _metrics_extra_kcal_to_ev(mae_kcal: float, rmse_kcal: float) -> Dict[str, float]:
    ev_per_kcalmol = 1.0 / float(23.06035)  # ~0.04336 eV per kcal/mol
    return {
        "mae_ev": mae_kcal * ev_per_kcalmol,
        "rmse_ev": rmse_kcal * ev_per_kcalmol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PhysNetJAX checkpoint on NPZ (energies, forces, dipoles).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PhysNet checkpoint root (directory containing epoch-* orbax runs), "
        "same as mmml physnet-md --checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="NPZ with R, Z, N, E, F (and optionally D / Dxyz / dipole if model predicts dipoles)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("physnet_evaluate_out"),
        help="Directory for metrics.json and optional plots (default: ./physnet_evaluate_out)",
    )
    parser.add_argument(
        "--natoms",
        type=int,
        default=None,
        help="Padded atom count (must match training). Default: inferred from NPZ Z/R width.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16). Remainder samples are skipped.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for batch shuffling (default: 0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="If set, evaluate at most this many structures (after shuffle split).",
    )
    parser.add_argument(
        "--subtract-atom-energies",
        action="store_true",
        help="Subtract atomic reference energies from E (same option as training data prep).",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract mean energy from E (training-style).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write parity plots (requires matplotlib).",
    )
    parser.add_argument(
        "--no-save-npz",
        action="store_true",
        help="Do not write predictions.npz (default: save).",
    )
    args = parser.parse_args()

    import jax
    import matplotlib

    matplotlib.use("Agg")

    if not args.checkpoint.exists():
        print(f"Error: checkpoint path not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"Error: data NPZ not found: {args.data}", file=sys.stderr)
        return 1

    data_path, tmp_created = _ensure_npz_with_N(args.data)
    try:
        natoms = (
            args.natoms
            if args.natoms is not None
            else _infer_natoms_from_npz(data_path)
        )
        n_total = _count_samples(data_path)
        n_eval = (
            n_total
            if args.num_samples is None
            else min(int(args.num_samples), n_total)
        )
        if n_eval < 1:
            print("Error: no samples to evaluate.", file=sys.stderr)
            return 1

        from mmml.physnetjax.physnetjax.restart.restart import (
            get_last,
            get_params_model,
        )
        from mmml.physnetjax.physnetjax.data.data import prepare_datasets
        from mmml.physnetjax.physnetjax.data.batches import prepare_batches_jit
        from mmml.physnetjax.physnetjax.analysis.analysis import plot_stats

        restart_path = get_last(str(args.checkpoint))
        params, model = get_params_model(str(restart_path), natoms=natoms)
        if model is None:
            print(
                "Error: checkpoint has no model_attributes; cannot rebuild EF model.",
                file=sys.stderr,
            )
            return 1

        key = jax.random.PRNGKey(args.seed)
        # Entire NPZ as the validation split (train split empty).
        _, valid_data = prepare_datasets(
            key,
            train_size=0,
            valid_size=n_eval,
            files=[str(data_path)],
            natoms=natoms,
            verbose=False,
            subtract_atom_energies=args.subtract_atom_energies,
            subtract_mean=args.subtract_mean,
        )

        required = ("R", "Z", "N", "E", "F")
        for k in required:
            if k not in valid_data:
                print(
                    f"Error: after loading, valid_data missing required key '{k}'.",
                    file=sys.stderr,
                )
                return 1

        _normalize_valid_data(valid_data)

        data_keys: List[str] = ["R", "Z", "F", "E", "N"]
        if getattr(model, "charges", False) and "D" in valid_data:
            data_keys.append("D")

        key2, _ = jax.random.split(key)
        try:
            batches = prepare_batches_jit(
                key2,
                valid_data,
                args.batch_size,
                data_keys=data_keys,
                num_atoms=natoms,
            )
        except ValueError as e:
            print(f"Error building batches: {e}", file=sys.stderr)
            return 1

        if not batches:
            print(
                "Error: no full batches (increase dataset size or lower --batch-size).",
                file=sys.stderr,
            )
            return 1

        print(
            f"\nEvaluating {len(batches)} batches × batch_size={args.batch_size} "
            f"(natoms={natoms}, checkpoint={restart_path.name})\n"
        )

        stats = plot_stats(
            batches,
            model,
            params,
            _set=f"PhysNet eval | {args.data.name}",
            do_kde=False,
            batch_size=args.batch_size,
            do_plot=args.plots,
        )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics: Dict[str, Any] = {
            "checkpoint": str(restart_path),
            "data": str(args.data.resolve()),
            "natoms": natoms,
            "n_batches": len(batches),
            "batch_size": args.batch_size,
            "energy": {
                "mae_kcal_mol": float(stats["E_mae"]),
                "rmse_kcal_mol": float(stats["E_rmse"]),
                **_metrics_extra_kcal_to_ev(
                    float(stats["E_mae"]), float(stats["E_rmse"])
                ),
            },
            "forces": {
                "mae_kcal_mol": float(stats["F_mae"]),
                "rmse_kcal_mol": float(stats["F_rmse"]),
                **_metrics_extra_kcal_to_ev(
                    float(stats["F_mae"]), float(stats["F_rmse"])
                ),
            },
            "n_params": int(stats["n_params"]),
        }
        if stats.get("D_mae") is not None:
            metrics["dipole"] = {
                "mae_e_bohr": float(stats["D_mae"]),
                "rmse_e_bohr": float(stats["D_rmse"]),
            }

        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {metrics_path}")

        if args.plots:
            import matplotlib.pyplot as plt

            fig_path = out_dir / "parity_plots.png"
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Wrote {fig_path}")

        if not args.no_save_npz:
            pred_path = out_dir / "predictions.npz"
            pred_kw: Dict[str, np.ndarray] = {
                "E_ref_kcal_mol": np.asarray(stats["Es"]),
                "E_pred_kcal_mol": np.asarray(stats["predEs"]),
                "F_ref_kcal_mol": np.asarray(stats["Fs"]),
                "F_pred_kcal_mol": np.asarray(stats["predFs"]),
            }
            if stats.get("D_mae") is not None:
                pred_kw["D_ref"] = np.asarray(stats["Ds"]).reshape(-1)
                pred_kw["D_pred"] = np.asarray(stats["predDs"]).reshape(-1)
            np.savez(pred_path, **pred_kw)
            print(f"Wrote {pred_path}")

        print(
            "\nSummary (energies/forces in kcal/mol; dipoles in e·Bohr when present):"
        )
        print(f"  Energy   MAE={stats['E_mae']:.6f}  RMSE={stats['E_rmse']:.6f}")
        print(f"  Forces   MAE={stats['F_mae']:.6f}  RMSE={stats['F_rmse']:.6f}")
        if stats.get("D_mae") is not None:
            print(f"  Dipole   MAE={stats['D_mae']:.6f}  RMSE={stats['D_rmse']:.6f}")

        return 0
    finally:
        if tmp_created:
            try:
                data_path.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
