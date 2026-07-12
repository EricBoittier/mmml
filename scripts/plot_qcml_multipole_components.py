#!/usr/bin/env python3
"""Plot QCML multipole parity/error diagnostics for each spherical (l, m) component."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mmml-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.analyze_qcml_multipoles import (
        build_predict_step,
        load_model,
        predict_bucketed,
        resolve_split_paths,
        select_single_shard_indices,
    )
    from scripts.train_qcml_multipoles import eligible_indices, restore_cache
except ModuleNotFoundError:
    from analyze_qcml_multipoles import (
        build_predict_step,
        load_model,
        predict_bucketed,
        resolve_split_paths,
        select_single_shard_indices,
    )
    from train_qcml_multipoles import eligible_indices, restore_cache


def component_metadata(max_degree: int = 3) -> list[dict[str, Any]]:
    rows = []
    offset = 0
    for degree in range(max_degree + 1):
        for component, order in enumerate(range(-degree, degree + 1)):
            rows.append(
                {
                    "index": offset + component,
                    "degree": degree,
                    "order": order,
                    "name": f"l{degree}_m{order:+d}",
                }
            )
        offset += 2 * degree + 1
    return rows


def load_scale_vector(path: Path | None, max_degree: int = 3) -> np.ndarray:
    if path is None:
        return np.ones(sum(2 * degree + 1 for degree in range(max_degree + 1)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    scale_payload = payload.get("scale", payload)
    values = []
    for degree in range(max_degree + 1):
        values.extend([float(scale_payload[f"l{degree}"])] * (2 * degree + 1))
    return np.asarray(values, dtype=np.float64)


def resolve_scale_json(path: Path | None, checkpoint: Path) -> Path | None:
    """Resolve scale JSON before expensive prediction work starts."""
    candidates = []
    if path is not None:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded
        candidates.extend(
            [
                expanded.with_name("target_scale.json"),
                expanded.with_name("target_rms.json"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                print(
                    f"Scale JSON not found at {expanded}; using {candidate} instead",
                    flush=True,
                )
                return candidate
        tried = "\n  ".join(str(candidate) for candidate in [expanded, *candidates])
        raise FileNotFoundError(
            "Could not find scale JSON before prediction. Tried:\n"
            f"  {tried}\n"
            "Use --scale-json <run_dir>/target_scale.json, omit --scale-json, "
            "or rerun without --normalize-by-scale."
        )

    run_dir = checkpoint.expanduser().parent
    for candidate in (run_dir / "target_scale.json", run_dir / "target_rms.json"):
        if candidate.exists():
            return candidate
    return None


def identity_limits(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = 0.04 * (high - low)
    return low - padding, high + padding


def select_plot_points(size: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or size <= max_points:
        return np.arange(size)
    return np.random.default_rng(seed).choice(size, size=max_points, replace=False)


def collect_predictions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model, params = load_model(args.checkpoint)
    shard_paths = resolve_split_paths(
        args.cache,
        args.checkpoint,
        args.split,
        args.validation_shards,
        args.test_shards,
        args.data_split,
    )
    all_predictions = []
    all_targets = []
    all_indices = []
    all_num_atoms = []
    remaining = args.max_structures
    predict_step = build_predict_step(model, params, args.batch_size)

    if shard_paths is None:
        cache = restore_cache(args.cache)
        indices = select_single_shard_indices(
            cache,
            args.split,
            args.validation_fraction,
            args.seed,
            args.max_atoms,
        )
        if remaining is not None:
            indices = indices[:remaining]
        shard_work = [(args.cache, cache, indices)]
    else:
        shard_work = []
        for shard_number, shard_path in enumerate(shard_paths, start=1):
            if remaining is not None and remaining <= 0:
                break
            print(f"Restoring shard {shard_number}/{len(shard_paths)}: {shard_path}")
            cache = restore_cache(shard_path)
            indices = eligible_indices(cache, args.max_atoms)
            if remaining is not None:
                indices = indices[:remaining]
                remaining -= len(indices)
            shard_work.append((shard_path, cache, indices))

    for shard_number, (shard_path, cache, indices) in enumerate(shard_work, start=1):
        if not len(indices):
            continue
        print(
            f"Predicting shard {shard_number}/{len(shard_work)}: "
            f"{shard_path} ({len(indices)} structures)"
        )
        prediction, ordered_indices = predict_bucketed(
            predict_step,
            cache,
            indices,
            args.batch_size,
            args.bucket_width,
        )
        all_predictions.append(prediction)
        all_targets.append(cache["multipoles"][ordered_indices].astype(np.float64))
        all_indices.append(ordered_indices)
        all_num_atoms.append(cache["atom_mask"][ordered_indices].sum(axis=1))

    if not all_indices:
        raise ValueError(f"The selected {args.split} split contains no eligible structures")
    return (
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_predictions, axis=0),
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_num_atoms, axis=0),
    )


def component_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    scale_vector: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for meta in component_metadata():
        index = meta["index"]
        scale = float(scale_vector[index])
        target_values = target[:, index]
        prediction_values = prediction[:, index]
        error = prediction_values - target_values
        normalized_error = error / scale
        centered_target = target_values - np.mean(target_values)
        centered_prediction = prediction_values - np.mean(prediction_values)
        denominator = np.linalg.norm(centered_target) * np.linalg.norm(centered_prediction)
        correlation = (
            float(np.dot(centered_target, centered_prediction) / denominator)
            if denominator
            else float("nan")
        )
        rows.append(
            {
                **meta,
                "scale": scale,
                "target_mean": float(np.mean(target_values)),
                "target_std": float(np.std(target_values)),
                "target_q01": float(np.quantile(target_values, 0.01)),
                "target_q99": float(np.quantile(target_values, 0.99)),
                "prediction_mean": float(np.mean(prediction_values)),
                "prediction_std": float(np.std(prediction_values)),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "normalized_mae": float(np.mean(np.abs(normalized_error))),
                "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
                "correlation": correlation,
            }
        )
    return rows


def write_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "component_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "component_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def plot_component_parity(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
    unit: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.scatter(target, prediction, s=5, alpha=0.25, rasterized=True)
    limits = identity_limits(target, prediction)
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axis.set(xlim=limits, ylim=limits, xlabel=f"Reference [{unit}]", ylabel=f"Prediction [{unit}]")
    axis.set_title(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_component_error(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
    unit: str,
) -> None:
    error = prediction - target
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].scatter(target, error, s=5, alpha=0.25, rasterized=True)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel(f"Reference [{unit}]")
    axes[0].set_ylabel(f"Prediction - reference [{unit}]")
    axes[0].set_title(f"{title}: error vs reference")
    axes[1].hist(error, bins=80, alpha=0.8)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel(f"Prediction - reference [{unit}]")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title}: error distribution")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_degree_grid(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    degree: int,
    plot_indices: np.ndarray,
    unit: str,
) -> None:
    metas = [meta for meta in component_metadata() if meta["degree"] == degree]
    columns = min(len(metas), 4)
    rows = int(np.ceil(len(metas) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, meta in zip(axes.flat, metas, strict=False):
        index = meta["index"]
        target_values = target[plot_indices, index]
        prediction_values = prediction[plot_indices, index]
        axis.scatter(target_values, prediction_values, s=4, alpha=0.25, rasterized=True)
        limits = identity_limits(target_values, prediction_values)
        axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
        axis.set(xlim=limits, ylim=limits, title=meta["name"])
        axis.set_xlabel(f"Reference [{unit}]")
        axis.set_ylabel(f"Prediction [{unit}]")
    for axis in axes.flat[len(metas):]:
        axis.axis("off")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_all_components(
    output_dir: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    plot_indices: np.ndarray,
    unit: str,
) -> None:
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    for meta in component_metadata():
        index = meta["index"]
        target_values = target[plot_indices, index]
        prediction_values = prediction[plot_indices, index]
        plot_component_parity(
            component_dir / f"xy_{meta['name']}.png",
            target_values,
            prediction_values,
            f"Spherical multipole {meta['name']}",
            unit,
        )
        plot_component_error(
            component_dir / f"error_{meta['name']}.png",
            target_values,
            prediction_values,
            f"Spherical multipole {meta['name']}",
            unit,
        )
    for degree in range(4):
        plot_degree_grid(
            output_dir / f"xy_grid_l{degree}.png",
            target,
            prediction,
            degree,
            plot_indices,
            unit,
        )


def write_npz(
    output_dir: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    indices: np.ndarray,
    num_atoms: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / "predictions.npz",
        target=target,
        prediction=prediction,
        dataset_index=indices,
        num_atoms=num_atoms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bucket-width", type=int, default=16)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="test")
    parser.add_argument("--data-split", type=Path)
    parser.add_argument("--validation-shards", type=int, default=2)
    parser.add_argument("--test-shards", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--max-plot-points", type=int, default=50000)
    parser.add_argument("--scale-json", type=Path)
    parser.add_argument("--normalize-by-scale", action="store_true")
    parser.add_argument("--unit", default="QCML native")
    parser.add_argument("--save-npz", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale_json = resolve_scale_json(args.scale_json, args.checkpoint)
    scale_vector = load_scale_vector(scale_json)
    if scale_json is not None:
        print(f"Using scale JSON: {scale_json}", flush=True)
    target, prediction, indices, num_atoms = collect_predictions(args)
    metrics = component_metrics(target, prediction, scale_vector)
    write_metrics(args.output_dir, metrics)

    plot_target = target / scale_vector if args.normalize_by_scale else target
    plot_prediction = prediction / scale_vector if args.normalize_by_scale else prediction
    unit = "normalized by target scale" if args.normalize_by_scale else args.unit
    plot_indices = select_plot_points(len(target), args.max_plot_points, args.seed)
    plot_all_components(args.output_dir, plot_target, plot_prediction, plot_indices, unit)
    if args.save_npz:
        write_npz(args.output_dir, target, prediction, indices, num_atoms)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "num_structures": int(len(target)),
        "num_plot_points": int(len(plot_indices)),
        "normalized_by_scale": bool(args.normalize_by_scale),
        "scale_json": str(scale_json) if scale_json else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote component plots to {args.output_dir}")


if __name__ == "__main__":
    main()
