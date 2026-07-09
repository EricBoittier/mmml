#!/usr/bin/env python3
"""Evaluate a QCML multipole checkpoint and generate numerical/visual reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mmml-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmml.data.orbax_shards import partition_shards
from mmml.models.multipoles import E3xMultipoleModel, irrep_blocks_to_traceless
try:
    from scripts.train_qcml_multipoles import (
        TrainConfig,
        bucket_indices,
        eligible_indices,
        iter_batches,
        iter_bucket_batches,
        make_batch,
        restore_cache,
        split_indices,
    )
except ModuleNotFoundError:
    from train_qcml_multipoles import (
        TrainConfig,
        bucket_indices,
        eligible_indices,
        iter_batches,
        iter_bucket_batches,
        make_batch,
        restore_cache,
        split_indices,
    )


TENSOR_KEYS = {
    0: "l0_monopole",
    1: "l1_dipole_tensor",
    2: "l2_quadrupole_tensor",
    3: "l3_octupole_tensor",
}


def load_model(checkpoint: Path) -> tuple[E3xMultipoleModel, Any]:
    """Restore model configuration and parameters from an epoch directory."""
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(TrainConfig)}
    model_config = {key: value for key, value in raw_config.items() if key in valid_fields}
    TrainConfig(**model_config)
    restored = ocp.PyTreeCheckpointer().restore(checkpoint)
    if "params" not in restored:
        raise KeyError(f"Checkpoint {checkpoint} does not contain params")
    return E3xMultipoleModel(**model_config), restored["params"]


def select_indices(
    size: int,
    split: str,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    if split == "all":
        return np.arange(size)
    training, validation = split_indices(size, validation_fraction, seed)
    return training if split == "train" else validation


def resolve_split_paths(
    cache: Path,
    checkpoint: Path,
    split: str,
    validation_shards: int,
    test_shards: int,
    data_split: Path | None,
) -> list[Path] | None:
    """Return shard paths for manifest caches, or None for single-shard caches."""
    split_path = data_split or checkpoint.parent / "data_split.json"
    if split_path.exists():
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        if split == "all":
            paths = split_data.get("train", []) + split_data.get("validation", []) + split_data.get("test", [])
        else:
            paths = split_data.get(split, [])
        if not paths:
            raise ValueError(f"No paths found for split={split!r} in {split_path}")
        return [Path(path) for path in paths]

    if (cache / "manifest.json").exists():
        partitions = partition_shards(
            cache,
            validation_shards=validation_shards,
            test_shards=test_shards,
        )
        if split == "all":
            return partitions["train"] + partitions["validation"] + partitions["test"]
        return partitions[split]
    return None


def select_single_shard_indices(
    cache: dict[str, np.ndarray],
    split: str,
    validation_fraction: float,
    seed: int,
    max_atoms: int | None,
) -> np.ndarray:
    eligible = eligible_indices(cache, max_atoms)
    if split == "all":
        return eligible
    training, validation = split_indices(len(eligible), validation_fraction, seed)
    if split == "train":
        return eligible[training]
    if split == "validation":
        return eligible[validation]
    raise ValueError("Single-shard cache has no held-out test split; use --split validation/all/train")


def predict(
    model: E3xMultipoleModel,
    params: Any,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Run fixed-shape batches and discard repeated final-batch padding."""
    predictions = []
    for batch_indices, example_mask in iter_batches(indices, batch_size):
        batch = make_batch(cache, batch_indices, example_mask)
        inputs = {
            key: batch[key]
            for key in (
                "positions",
                "atomic_numbers",
                "charge",
                "spin",
                "dst_idx",
                "src_idx",
                "batch_segments",
                "atom_mask",
                "edge_mask",
            )
        }
        output = model.apply(
            {"params": params},
            **inputs,
            batch_size=batch_size,
        )
        valid_count = int(example_mask.sum())
        predictions.append(np.asarray(output["multipoles"][:valid_count]))
    return np.concatenate(predictions, axis=0)


def predict_bucketed(
    model: E3xMultipoleModel,
    params: Any,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    bucket_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run atom-count bucketed batches and discard repeated final-batch padding."""
    predictions = []
    ordered_indices = []
    buckets = bucket_indices(cache, indices, bucket_width)
    for batch_indices, example_mask, max_atoms in iter_bucket_batches(
        buckets,
        batch_size,
    ):
        batch = make_batch(cache, batch_indices, example_mask, max_atoms)
        inputs = {
            key: batch[key]
            for key in (
                "positions",
                "atomic_numbers",
                "charge",
                "spin",
                "dst_idx",
                "src_idx",
                "batch_segments",
                "atom_mask",
                "edge_mask",
            )
        }
        output = model.apply(
            {"params": params},
            **inputs,
            batch_size=batch_size,
        )
        valid_count = int(example_mask.sum())
        predictions.append(np.asarray(output["multipoles"][:valid_count]))
        ordered_indices.append(batch_indices[:valid_count])
    return np.concatenate(predictions, axis=0), np.concatenate(ordered_indices, axis=0)


def _degree_blocks(values: np.ndarray) -> dict[int, np.ndarray]:
    blocks = {}
    start = 0
    for degree in range(4):
        width = 2 * degree + 1
        blocks[degree] = values[:, start : start + width]
        start += width
    return blocks


def _cartesian_blocks(values: np.ndarray) -> dict[int, np.ndarray]:
    converted = irrep_blocks_to_traceless(jnp.asarray(values), max_degree=3)
    return {
        degree: np.asarray(converted[key]).reshape(values.shape[0], -1)
        for degree, key in TENSOR_KEYS.items()
    }


def error_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    num_atoms: np.ndarray,
    scales: dict[int, float],
    units: dict[int, str],
) -> dict[str, Any]:
    """Compute spherical-component and Cartesian-tensor errors by degree."""
    target_spherical = _degree_blocks(target)
    prediction_spherical = _degree_blocks(prediction)
    target_cartesian = _cartesian_blocks(target)
    prediction_cartesian = _cartesian_blocks(prediction)
    report: dict[str, Any] = {}

    for degree in range(4):
        scale = scales[degree]
        degree_report: dict[str, Any] = {"unit": units[degree], "scale": scale}
        for representation, references, estimates in (
            ("spherical_traceless", target_spherical, prediction_spherical),
            ("cartesian_traceless", target_cartesian, prediction_cartesian),
        ):
            error = (estimates[degree] - references[degree]) * scale
            norm_error = np.linalg.norm(error, axis=-1)
            degree_report[representation] = {
                "component_mae": float(np.mean(np.abs(error))),
                "component_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "tensor_norm_mae": float(np.mean(norm_error)),
                "tensor_norm_rmse": float(np.sqrt(np.mean(np.square(norm_error)))),
                "per_atom_tensor_norm_mae": float(np.mean(norm_error / num_atoms)),
                "max_tensor_norm_error": float(np.max(norm_error)),
            }
        report[f"l{degree}"] = degree_report
    return report


def write_per_structure_csv(
    path: Path,
    indices: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    num_atoms: np.ndarray,
    scales: dict[int, float],
) -> None:
    target_blocks = _degree_blocks(target)
    prediction_blocks = _degree_blocks(prediction)
    fieldnames = ["dataset_index", "num_atoms"]
    for degree in range(4):
        fieldnames.extend(
            [
                f"l{degree}_target_norm",
                f"l{degree}_prediction_norm",
                f"l{degree}_error_norm",
                f"l{degree}_error_norm_per_atom",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, dataset_index in enumerate(indices):
            row: dict[str, Any] = {
                "dataset_index": int(dataset_index),
                "num_atoms": int(num_atoms[row_index]),
            }
            for degree in range(4):
                scale = scales[degree]
                target_norm = np.linalg.norm(target_blocks[degree][row_index]) * scale
                prediction_norm = np.linalg.norm(prediction_blocks[degree][row_index]) * scale
                error_norm = (
                    np.linalg.norm(
                        prediction_blocks[degree][row_index]
                        - target_blocks[degree][row_index]
                    )
                    * scale
                )
                row.update(
                    {
                        f"l{degree}_target_norm": target_norm,
                        f"l{degree}_prediction_norm": prediction_norm,
                        f"l{degree}_error_norm": error_norm,
                        f"l{degree}_error_norm_per_atom": error_norm / num_atoms[row_index],
                    }
                )
            writer.writerow(row)


def _identity_limits(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = 0.04 * (high - low)
    return low - padding, high + padding


def plot_scatter(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
    unit: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    for component in range(target.shape[1]):
        axis.scatter(
            target[:, component],
            prediction[:, component],
            s=7,
            alpha=0.35,
            label=f"c{component}",
        )
    limits = _identity_limits(target, prediction)
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axis.set(xlim=limits, ylim=limits, xlabel=f"Reference [{unit}]", ylabel=f"Prediction [{unit}]")
    axis.set_title(title)
    if target.shape[1] <= 7:
        axis.legend(markerscale=2, fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_error_distributions(
    path: Path,
    errors: dict[int, np.ndarray],
    num_atoms: np.ndarray,
    units: dict[int, str],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for degree, axis in enumerate(axes.flat):
        error_norm = np.linalg.norm(errors[degree], axis=-1)
        axis.hist(error_norm, bins=50, alpha=0.65, label="molecule")
        axis.hist(error_norm / num_atoms, bins=50, alpha=0.65, label="per atom")
        axis.set_title(f"l={degree}")
        axis.set_xlabel(f"Error norm [{units[degree]}]")
        axis.set_ylabel("Count")
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_error_vs_atoms(
    path: Path,
    errors: dict[int, np.ndarray],
    num_atoms: np.ndarray,
    units: dict[int, str],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for degree, axis in enumerate(axes.flat):
        error_norm = np.linalg.norm(errors[degree], axis=-1)
        axis.scatter(num_atoms, error_norm, s=8, alpha=0.35)
        unique_atoms = np.unique(num_atoms)
        means = [np.mean(error_norm[num_atoms == count]) for count in unique_atoms]
        axis.plot(unique_atoms, means, color="black", linewidth=1.5, label="mean")
        axis.set_title(f"l={degree}")
        axis.set_xlabel("Number of atoms")
        axis.set_ylabel(f"Error norm [{units[degree]}]")
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_report(
    output_dir: Path,
    indices: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    num_atoms: np.ndarray,
    scales: dict[int, float],
    units: dict[int, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = error_metrics(target, prediction, num_atoms, scales, units)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_per_structure_csv(
        output_dir / "per_structure_errors.csv",
        indices,
        target,
        prediction,
        num_atoms,
        scales,
    )

    spherical_target = _degree_blocks(target)
    spherical_prediction = _degree_blocks(prediction)
    cartesian_target = _cartesian_blocks(target)
    cartesian_prediction = _cartesian_blocks(prediction)
    spherical_errors = {}
    for degree in range(4):
        scale = scales[degree]
        spherical_errors[degree] = (
            spherical_prediction[degree] - spherical_target[degree]
        ) * scale
        plot_scatter(
            output_dir / f"scatter_spherical_l{degree}.png",
            spherical_target[degree] * scale,
            spherical_prediction[degree] * scale,
            f"Spherical traceless multipole, l={degree}",
            units[degree],
        )
        plot_scatter(
            output_dir / f"scatter_cartesian_l{degree}.png",
            cartesian_target[degree] * scale,
            cartesian_prediction[degree] * scale,
            f"Cartesian traceless tensor, l={degree}",
            units[degree],
        )
    plot_error_distributions(
        output_dir / "error_distributions.png",
        spherical_errors,
        num_atoms,
        units,
    )
    plot_error_vs_atoms(
        output_dir / "error_vs_num_atoms.png",
        spherical_errors,
        num_atoms,
        units,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("all", "train", "validation"), default="validation")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    for degree in range(4):
        parser.add_argument(f"--scale-l{degree}", type=float, default=1.0)
        parser.add_argument(f"--unit-l{degree}", default="QCML native")
    args = parser.parse_args()

    cache = restore_cache(args.cache)
    model, params = load_model(args.checkpoint)
    indices = select_indices(
        cache["R"].shape[0],
        args.split,
        args.validation_fraction,
        args.seed,
    )
    if not len(indices):
        raise ValueError(f"The selected {args.split} split is empty")
    prediction = predict(model, params, cache, indices, args.batch_size)
    target = cache["multipoles"][indices].astype(np.float32)
    num_atoms = cache["atom_mask"][indices].sum(axis=1)
    scales = {degree: getattr(args, f"scale_l{degree}") for degree in range(4)}
    units = {degree: getattr(args, f"unit_l{degree}") for degree in range(4)}
    metrics = generate_report(
        args.output_dir,
        indices,
        target,
        prediction,
        num_atoms,
        scales,
        units,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote report to {args.output_dir}")


if __name__ == "__main__":
    main()
