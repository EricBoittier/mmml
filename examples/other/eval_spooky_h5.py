#!/usr/bin/env python3
"""
Evaluate a spooky PhysNetJAX checkpoint on qcell-style HDF5 data.

Uses the same loaders and random split as :mod:`train_spooky_h5` when you pass
matching ``--train-size``, ``--valid-size``, and ``--seed`` (default 42). Energy
and force metrics use the **same units as in the HDF5** targets (no conversion).

Examples
--------
  # Validation split (same split as training when sizes/seed match train_spooky_h5.py)
  python examples/other/eval_spooky_h5.py \\
    --checkpoint ckpts_spooky_h5/final_params \\
    --data-dir /path/to/qcell \\
    --train-size 270000 --valid-size 1000

  # Single file, evaluate training split only
  python examples/other/eval_spooky_h5.py \\
    --checkpoint ckpts_spooky_h5/final_params \\
    --filepath /path/to/qcell_dimers.h5 \\
    --eval-on train --train-size 1000 --valid-size 200

  # Entire dataset (no train/valid partition; one ``load_h5`` pass)
  python examples/other/eval_spooky_h5.py \\
    --checkpoint latest --checkpoint-root ckpts_spooky_h5 \\
    --filepath data.h5 --eval-on all

Prerequisites: jax, flax, h5py, matplotlib, orbax, numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp

from mmml.models.physnetjax.physnetjax.data.read_h5 import (
    _detect_natoms,
    load_h5,
    prepare_h5_datasets,
)
from mmml.models.physnetjax.physnetjax.models.spooky_model import EF as SpookyEF
from mmml.models.physnetjax.physnetjax.training.spooky_training import (
    build_spooky_batch_from_padded_arrays,
    forward_spooky_batch,
    restart_params_only,
)

DEFAULT_CHECKPOINT_ROOT = Path("ckpts_spooky_h5").resolve()
OUTPUT_DIR = Path("eval_spooky_h5_out").resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate spooky PhysNetJAX on qcell HDF5 (parity metrics + plots)."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Orbax checkpoint dir (e.g. .../final_params), or 'latest' (see --checkpoint-root).",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help='When --checkpoint is "latest", load <checkpoint-root>/final_params (default: ckpts_spooky_h5).',
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--filepath",
        type=str,
        nargs="+",
        help="Path(s) to HDF5 file(s).",
    )
    grp.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing qcell_*.h5 files.",
    )
    p.add_argument("--train-size", type=int, default=270_000)
    p.add_argument("--valid-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="PRNG seed for prepare_h5_datasets split.")
    p.add_argument(
        "--eval-on",
        type=str,
        choices=("valid", "train", "all"),
        default="valid",
        help="Which split to score: valid / train (from prepare_h5_datasets), or all loaded structures.",
    )
    p.add_argument("--natoms", type=int, default=None)
    p.add_argument("--max-structures", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="After choosing the split, evaluate at most this many structures (first N).",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Include a final batch smaller than --batch-size (slower; no JIT).",
    )
    p.add_argument(
        "--charge-filter",
        type=float,
        default=None,
        help="If set, only include structures with this total charge.",
    )
    p.add_argument("--energy-key", type=str, default="formation_energy")
    p.add_argument("--force-key", type=str, default="total_forces")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG figures.",
    )
    p.add_argument(
        "--plot-max-points",
        type=int,
        default=50_000,
        help="Max points for dense scatter / hexbin (subsample beyond this).",
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write predictions.npz: E, F (masked flat), Z, R, Q, S (per structure scored).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_filepaths(args: argparse.Namespace) -> list[str]:
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
        paths = sorted(data_dir.glob("qcell_*.h5"))
        if not paths:
            raise FileNotFoundError(f"No qcell_*.h5 files found in {data_dir}")
        return [str(p) for p in paths]
    fps = [Path(p) for p in args.filepath]
    if len(fps) == 1:
        return [str(fps[0])]
    return [str(p) for p in fps]


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint == "latest":
        root = Path(args.checkpoint_root).resolve()
        p = root / "final_params"
        if not p.is_dir():
            raise FileNotFoundError(
                f'--checkpoint latest expects {p} to exist (checkpoint root: {root}).'
            )
        return p
    p = Path(args.checkpoint)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _slice_data(data: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    out = {}
    for k, v in data.items():
        out[k] = np.asarray(v)[:n]
    return out


def _print_dataset_stats(name: str, data: dict[str, np.ndarray]) -> None:
    z = np.asarray(data["Z"])
    r = np.asarray(data["R"])
    e = np.asarray(data["E"]).reshape(-1)
    f = np.asarray(data["F"])
    n_struct = len(e)
    n_atoms_real = np.sum(z > 0, axis=1)

    print(f"\n--- Dataset statistics ({name}, n={n_struct}) ---")
    print(f"  natoms (padding width): {z.shape[1]}")
    print(
        f"  Real atoms per structure: min={int(n_atoms_real.min())} "
        f"max={int(n_atoms_real.max())} mean={float(n_atoms_real.mean()):.2f}"
    )
    print(
        f"  Energy E: min={e.min():.6g} max={e.max():.6g} "
        f"mean={e.mean():.6g} std={e.std():.6g}"
    )
    mask = z > 0
    f_real = f[mask]
    print(
        f"  Forces (real atoms, all components): min={f_real.min():.6g} "
        f"max={f_real.max():.6g} mean={f_real.mean():.6g} std={f_real.std():.6g}"
    )
    fnorm = np.linalg.norm(f, axis=-1)[mask]
    print(f"  |F| per atom: mean={float(fnorm.mean()):.6g} std={float(fnorm.std()):.6g}")

    if "Q" in data:
        q = np.asarray(data["Q"]).reshape(-1)
        print(f"  Total charge Q: min={q.min():.6g} max={q.max():.6g} unique≈{len(np.unique(q))}")
    if "S" in data:
        s = np.asarray(data["S"]).reshape(-1)
        print(f"  Spin multiplicity S: min={s.min():.6g} max={s.max():.6g} unique≈{len(np.unique(s))}")


def _metrics_vec(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse}


def _r2_energy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-30:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2:
        return float("nan")
    cx = x - x.mean()
    cy = y - y.mean()
    denom = float(np.sqrt(np.sum(cx**2) * np.sum(cy**2)))
    if denom < 1e-30:
        return float("nan")
    return float(np.sum(cx * cy) / denom)


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filepath_arg = _resolve_filepaths(args)
    print("Files:", filepath_arg)

    key = jax.random.PRNGKey(int(args.seed))
    natoms_opt = args.natoms

    if args.eval_on == "all":
        if natoms_opt is None:
            natoms_opt = _detect_natoms(
                filepath_arg,
                max_structures=args.max_structures,
                verbose=args.verbose,
            )
        eval_data = load_h5(
            filepath=filepath_arg,
            natoms=natoms_opt,
            energy_key=args.energy_key,
            force_key=args.force_key,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            cache=True,
            verbose=args.verbose,
        )
        natoms = natoms_opt
        split_label = "all"
    else:
        train_data, valid_data, natoms = prepare_h5_datasets(
            key,
            filepath=filepath_arg,
            train_size=args.train_size,
            valid_size=args.valid_size,
            natoms=natoms_opt,
            energy_key=args.energy_key,
            force_key=args.force_key,
            max_structures=args.max_structures,
            charge_filter=args.charge_filter,
            verbose=args.verbose,
        )
        eval_data = valid_data if args.eval_on == "valid" else train_data
        split_label = args.eval_on

    n_avail = len(eval_data["R"])
    if args.num_samples is not None:
        n_use = min(int(args.num_samples), n_avail)
        eval_data = _slice_data(eval_data, n_use)
    n_eval = len(eval_data["R"])
    if n_eval < 1:
        print("Error: no structures to evaluate.", file=sys.stderr)
        return 1

    _print_dataset_stats(f"{split_label} split", eval_data)

    ckpt_path = _resolve_checkpoint_path(args)
    print(f"\nCheckpoint: {ckpt_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    restored_params, restored_config, _s, _e, _c = restart_params_only(ckpt_path, checkpointer)
    if restored_config is None:
        raise ValueError(f"Checkpoint at {ckpt_path} has no 'config'; cannot rebuild SpookyEF.")

    def _to_native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    model_kwargs = {k: _to_native(v) for k, v in restored_config.items()}
    model = SpookyEF(**model_kwargs)
    if int(model.natoms) != int(natoms):
        raise ValueError(
            f"Checkpoint natoms={model.natoms} does not match dataset natoms={natoms}."
        )

    z = np.asarray(eval_data["Z"], dtype=np.int32)
    r = np.asarray(eval_data["R"], dtype=np.float32)
    e = np.asarray(eval_data["E"], dtype=np.float32)
    f = np.asarray(eval_data["F"], dtype=np.float32)
    q = np.asarray(eval_data["Q"], dtype=np.float32)
    s = np.asarray(eval_data["S"], dtype=np.float32)

    bs = args.batch_size
    n_full = (n_eval // bs) * bs
    n_skipped = n_eval - n_full if not args.include_partial else 0
    include_partial = bool(args.include_partial)
    if n_full < n_eval and not include_partial and n_eval > 0:
        if n_eval < bs:
            include_partial = True
            n_skipped = 0
            print(
                f"Note: n_eval={n_eval} < batch_size={bs}; evaluating in one partial batch.",
                file=sys.stderr,
            )
        else:
            print(
                f"Note: dropping last {n_skipped} structures (use --include-partial to score all).",
                file=sys.stderr,
            )
    include_partial_eff = include_partial

    e_true_all: list[np.ndarray] = []
    e_pred_all: list[np.ndarray] = []
    f_true_all: list[np.ndarray] = []
    f_pred_all: list[np.ndarray] = []

    def run_batch(i0: int, i1: int) -> None:
        sl = slice(i0, i1)
        batch = build_spooky_batch_from_padded_arrays(
            z[sl],
            r[sl],
            e[sl],
            f[sl],
            q[sl].flatten(),
            s[sl].flatten(),
        )
        out = forward_spooky_batch(model, restored_params, batch)
        e_pred = np.asarray(out["energy"], dtype=np.float64).reshape(-1)
        f_pred = np.asarray(out["forces"], dtype=np.float64)
        e_ref = np.asarray(batch["E"], dtype=np.float64).reshape(-1)
        f_ref = np.asarray(batch["F"], dtype=np.float64)
        am = np.asarray(batch["atom_mask"], dtype=np.float64) > 0.5
        e_true_all.append(e_ref)
        e_pred_all.append(e_pred)
        f_true_all.append(f_ref[am])
        f_pred_all.append(f_pred[am])

    for start in range(0, n_full, bs):
        run_batch(start, start + bs)

    if include_partial and n_full < n_eval:
        run_batch(n_full, n_eval)

    e_t = np.concatenate(e_true_all)
    e_p = np.concatenate(e_pred_all)
    f_t = np.concatenate(f_true_all)
    f_p = np.concatenate(f_pred_all)

    m_e = _metrics_vec(e_t, e_p)
    m_f = _metrics_vec(f_t, f_p)
    r2 = _r2_energy(e_t, e_p)
    pr = _pearson_r(e_t, e_p)

    print("\n--- Model vs reference (same units as HDF5) ---")
    print(f"  Energy   MAE={m_e['mae']:.8g}  RMSE={m_e['rmse']:.8g}  R²={r2:.6f}  r={pr:.6f}")
    print(f"  Forces (masked components) MAE={m_f['mae']:.8g}  RMSE={m_f['rmse']:.8g}")
    if not include_partial_eff and n_skipped > 0:
        print(f"  Note: dropped last {n_skipped} structures (use --include-partial to include).")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "checkpoint": str(ckpt_path),
        "files": filepath_arg,
        "split": split_label,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "seed": args.seed,
        "natoms": int(natoms),
        "n_evaluated": int(n_eval),
        "n_scored_structures": int(e_t.shape[0]),
        "n_scored_force_components": int(f_t.shape[0]),
        "batch_size": bs,
        "include_partial": bool(include_partial_eff),
        "n_skipped_remainder": int(n_skipped),
        "units_note": "Energy and forces in the same units as HDF5 targets (no conversion).",
        "energy": {**m_e, "r2": r2, "pearson_r": pr},
        "forces": m_f,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {metrics_path}")

    if args.save_predictions:
        pred_path = out_dir / "predictions.npz"
        n_scored = int(e_t.shape[0])
        np.savez(
            pred_path,
            E_ref=e_t,
            E_pred=e_p,
            F_ref=f_t,
            F_pred=f_p,
            Z=z[:n_scored],
            R=r[:n_scored],
            Q=q[:n_scored],
            S=s[:n_scored],
        )
        print(f"Wrote {pred_path}")

    if not args.no_plots:
        pmax = int(args.plot_max_points)

        def _subsample(n: int) -> np.ndarray:
            if n <= pmax:
                return np.arange(n, dtype=np.int64)
            rng = np.random.default_rng(args.seed)
            return rng.choice(n, size=pmax, replace=False)

        # Energy parity
        fig_e, ax_e = plt.subplots(figsize=(6, 6))
        idx = _subsample(len(e_t))
        et, ep = e_t[idx], e_p[idx]
        lo = min(et.min(), ep.min())
        hi = max(et.max(), ep.max())
        ax_e.scatter(et, ep, s=8, alpha=0.35, c="C0")
        ax_e.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        ax_e.set_aspect("equal", adjustable="box")
        ax_e.set_xlabel("Reference E (HDF5 units)")
        ax_e.set_ylabel("Predicted E (HDF5 units)")
        ax_e.set_title(f"Energy parity (n={len(e_t)})")
        t = f"MAE={m_e['mae']:.4g}\nRMSE={m_e['rmse']:.4g}\nR²={r2:.4f}"
        ax_e.text(0.05, 0.95, t, transform=ax_e.transAxes, va="top", fontsize=9)
        fig_e.tight_layout()
        p_e = out_dir / "parity_energy.png"
        fig_e.savefig(p_e, dpi=200, bbox_inches="tight")
        plt.close(fig_e)
        print(f"Wrote {p_e}")

        # Forces parity (subsample points for display)
        nf = f_t.shape[0]
        j = _subsample(nf)
        ft, fp = f_t[j], f_p[j]
        fig_f, ax_f = plt.subplots(figsize=(6, 6))
        flo = min(ft.min(), fp.min())
        fhi = max(ft.max(), fp.max())
        ax_f.scatter(ft, fp, s=1, alpha=0.2, c="C1")
        ax_f.plot([flo, fhi], [flo, fhi], color="gray", ls="--", lw=1)
        ax_f.set_aspect("equal", adjustable="box")
        ax_f.set_xlabel("Reference F component (HDF5 units)")
        ax_f.set_ylabel("Predicted F component (HDF5 units)")
        ax_f.set_title(f"Forces parity (masked components, n={f_t.size})")
        t = f"MAE={m_f['mae']:.4g}\nRMSE={m_f['rmse']:.4g}"
        ax_f.text(0.05, 0.95, t, transform=ax_f.transAxes, va="top", fontsize=9)
        fig_f.tight_layout()
        p_f = out_dir / "parity_forces.png"
        fig_f.savefig(p_f, dpi=200, bbox_inches="tight")
        plt.close(fig_f)
        print(f"Wrote {p_f}")

        # |ΔF| vs |F_ref|
        abs_ref = np.linalg.norm(f_t.reshape(-1, 3), axis=1)
        abs_err = np.linalg.norm((f_p - f_t).reshape(-1, 3), axis=1)
        nper = abs_ref.shape[0]
        k = _subsample(nper)
        fig_a, ax_a = plt.subplots(figsize=(6, 5))
        ax_a.scatter(abs_ref[k], abs_err[k], s=2, alpha=0.25, c="C2")
        ax_a.set_xlabel("|F_ref| per atom (HDF5 units)")
        ax_a.set_ylabel("|F_pred − F_ref| per atom")
        ax_a.set_title("Force error vs |F_ref|")
        fig_a.tight_layout()
        p_a = out_dir / "forces_abs_error.png"
        fig_a.savefig(p_a, dpi=200, bbox_inches="tight")
        plt.close(fig_a)
        print(f"Wrote {p_a}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
