"""Evaluate a KerNN JSON checkpoint on NPZ test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mmml.models.kernnn.checkpoint import load_checkpoint
from mmml.models.kernnn.model import energy_and_forces
from mmml.utils.cli_args import exit_if_unknown_long_options

EV_TO_KCAL_MOL = 23.060541945

_EVAL_DEFAULTS = {
    "checkpoint": "artifacts/kernnn/best.json",
    "data": "data.npz",
    "output_dir": "artifacts/kernnn/eval",
    "split": "test",
    "seed": 42,
    "ntrain": 3200,
    "nvalid": 400,
    "batch_size": 64,
}


def build_parser() -> argparse.ArgumentParser:
    d = _EVAL_DEFAULTS
    p = argparse.ArgumentParser(description="Evaluate KerNN checkpoint (E/F metrics)")
    p.add_argument("--checkpoint", type=str, default=d["checkpoint"])
    p.add_argument(
        "--data",
        type=str,
        default=d["data"],
        help="NPZ with R, E, F (use --split all for a dedicated test NPZ)",
    )
    p.add_argument("--output-dir", type=str, default=d["output_dir"])
    p.add_argument(
        "--split",
        type=str,
        default=d["split"],
        choices=("train", "valid", "test", "all"),
        help="Which split to evaluate (seed/ntrain/nvalid define the split; "
        "use 'all' for a dedicated test NPZ)",
    )
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--ntrain", type=int, default=d["ntrain"])
    p.add_argument("--nvalid", type=int, default=d["nvalid"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument(
        "--split-json",
        type=str,
        default=None,
        help="Optional data_split.json from training (overrides seed/ntrain/nvalid)",
    )
    return p


def get_args(argv: list[str] | None = None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    exit_if_unknown_long_options(unknown, prog="mmml kernnn-evaluate")
    return args


def _resolve_indices(args, ndata: int) -> np.ndarray:
    if args.split_json:
        split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        key = {
            "train": "idx_train",
            "valid": "idx_valid",
            "test": "idx_test",
            "all": None,
        }[args.split]
        if key is None:
            return np.arange(ndata)
        return np.asarray(split[key], dtype=np.int64)

    if args.split == "all":
        return np.arange(ndata)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(ndata)
    ntrain = int(args.ntrain)
    nvalid = int(args.nvalid)
    if args.split == "train":
        return idx[:ntrain]
    if args.split == "valid":
        return idx[ntrain : ntrain + nvalid]
    return idx[ntrain + nvalid :]


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = pred - ref
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if ref.size > 1 and np.std(ref) > 0:
        r2 = float(1.0 - np.sum(err**2) / np.sum((ref - np.mean(ref)) ** 2))
    else:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    params, config, stats, metadata = load_checkpoint(args.checkpoint)
    data = np.load(args.data)
    positions = np.asarray(data["R"], dtype=np.float32)
    energies = np.asarray(data["E"], dtype=np.float32).reshape(-1)
    forces = np.asarray(data["F"], dtype=np.float32)
    if positions.shape[1] != int(config.n_atoms):
        raise ValueError(
            f"checkpoint expects {config.n_atoms} atoms "
            f"(scheme={config.distance_scheme}); data has R shape {positions.shape}"
        )
    ndata = positions.shape[0]
    indices = _resolve_indices(args, ndata)

    pos = jnp.asarray(positions[indices])
    e_ref = np.asarray(energies[indices])
    f_ref = np.asarray(forces[indices])

    batch_size = int(args.batch_size)
    e_preds = []
    f_preds = []
    for start in range(0, len(indices), batch_size):
        sl = slice(start, start + batch_size)
        e_b, f_b = energy_and_forces(params, pos[sl], stats, config=config)
        e_preds.append(np.asarray(e_b))
        f_preds.append(np.asarray(f_b))
    e_pred = np.concatenate(e_preds, axis=0)
    f_pred = np.concatenate(f_preds, axis=0)

    e_m = _metrics(e_pred, e_ref)
    f_m = _metrics(f_pred.reshape(-1), f_ref.reshape(-1))

    report = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "split": args.split,
        "n": int(len(indices)),
        "energy_eV": e_m,
        "energy_kcal_mol": {
            k: (v * EV_TO_KCAL_MOL if k != "r2" else v) for k, v in e_m.items()
        },
        "forces_eV_A": f_m,
        "forces_kcal_mol_A": {
            k: (v * EV_TO_KCAL_MOL if k != "r2" else v) for k, v in f_m.items()
        },
        "metadata": metadata,
        "config": config.to_dict(),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Energy scatter (kcal/mol)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        e_ref * EV_TO_KCAL_MOL,
        e_pred * EV_TO_KCAL_MOL,
        s=8,
        alpha=0.5,
        edgecolors="none",
    )
    lo = min(e_ref.min(), e_pred.min()) * EV_TO_KCAL_MOL
    hi = max(e_ref.max(), e_pred.max()) * EV_TO_KCAL_MOL
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Reference E (kcal/mol)")
    ax.set_ylabel("KerNN E (kcal/mol)")
    ax.set_title(
        f"KerNN energy  MAE={report['energy_kcal_mol']['mae']:.3f} kcal/mol"
    )
    fig.tight_layout()
    fig.savefig(out / "energy_scatter.png", dpi=200)
    plt.close(fig)

    # Force component scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        f_ref.reshape(-1) * EV_TO_KCAL_MOL,
        f_pred.reshape(-1) * EV_TO_KCAL_MOL,
        s=4,
        alpha=0.3,
        edgecolors="none",
    )
    flo = min(f_ref.min(), f_pred.min()) * EV_TO_KCAL_MOL
    fhi = max(f_ref.max(), f_pred.max()) * EV_TO_KCAL_MOL
    ax.plot([flo, fhi], [flo, fhi], "k--", lw=1)
    ax.set_xlabel("Reference F (kcal/mol/Å)")
    ax.set_ylabel("KerNN F (kcal/mol/Å)")
    ax.set_title(
        f"KerNN forces  MAE={report['forces_kcal_mol_A']['mae']:.3f} kcal/mol/Å"
    )
    fig.tight_layout()
    fig.savefig(out / "force_scatter.png", dpi=200)
    plt.close(fig)

    np.savez_compressed(
        out / "predictions.npz",
        E_ref=e_ref,
        E_pred=e_pred,
        F_ref=f_ref,
        F_pred=f_pred,
        indices=indices,
    )

    print(json.dumps(report["energy_kcal_mol"], indent=2))
    print(json.dumps(report["forces_kcal_mol_A"], indent=2))
    print(f"wrote {out}")
    return report


def main(args=None):
    if args is None:
        args = get_args()
    return evaluate(args)


if __name__ == "__main__":
    main()
    sys.exit(0)
