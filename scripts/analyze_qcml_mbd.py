#!/usr/bin/env python3
"""Evaluate a QCML MBD checkpoint against cached energy, force, C6, and alpha targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
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
from mmml.models.mbd import E3xMBDModel, mbd_energy_and_forces

try:
    from scripts.train_qcml_mbd import (
        MBDTrainConfig,
        bucket_indices,
        eligible_indices,
        iter_bucket_batches,
        make_batch,
        model_inputs,
        restore_cache,
        split_indices,
    )
except ModuleNotFoundError:
    from train_qcml_mbd import (
        MBDTrainConfig,
        bucket_indices,
        eligible_indices,
        iter_bucket_batches,
        make_batch,
        model_inputs,
        restore_cache,
        split_indices,
    )


HARTREE_TO_KCAL_MOL = 627.5094740631
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_AU_TO_KCAL_MOL_ANGSTROM = HARTREE_TO_KCAL_MOL / BOHR_TO_ANGSTROM


def load_model(checkpoint: Path) -> tuple[E3xMBDModel, Any]:
    """Restore model configuration and parameters from an epoch directory."""
    config_path = checkpoint / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(MBDTrainConfig)}
    model_config = {key: value for key, value in raw_config.items() if key in valid_fields}
    MBDTrainConfig(**model_config)
    restored = ocp.PyTreeCheckpointer().restore(checkpoint)
    if "params" not in restored:
        raise KeyError(f"Checkpoint {checkpoint} does not contain params")
    return E3xMBDModel(**model_config), restored["params"]


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


def single_shard_indices(
    cache: dict[str, np.ndarray],
    split: str,
    validation_fraction: float,
    seed: int,
    max_atoms: int | None,
) -> np.ndarray:
    eligible = eligible_indices(cache, max_atoms)
    if split == "all":
        return eligible
    training_selection, validation_selection = split_indices(
        len(eligible),
        validation_fraction,
        seed,
    )
    if split == "train":
        return eligible[training_selection]
    if split == "validation":
        return eligible[validation_selection]
    raise ValueError("Single-shard cache has no held-out test split; use --split validation/all/train")


def predict_shard(
    model: E3xMBDModel,
    params: Any,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    bucket_width: int,
) -> dict[str, np.ndarray]:
    """Run fixed-shape bucketed batches and discard repeated final-batch padding."""
    predictions: dict[str, list[np.ndarray]] = {
        "energy": [],
        "forces": [],
        "c6": [],
        "alpha": [],
        "indices": [],
        "num_atoms": [],
    }

    @jax.jit
    def predict_step(batch):
        output, forces = mbd_energy_and_forces(model, params, **model_inputs(batch))
        return output, forces

    buckets = bucket_indices(cache, indices, bucket_width)
    for batch_indices, example_mask, max_atoms in iter_bucket_batches(
        buckets,
        batch_size,
    ):
        batch = make_batch(cache, batch_indices, example_mask, max_atoms)
        output, forces = predict_step(batch)
        valid_count = int(example_mask.sum())
        atom_mask = np.asarray(batch["atom_mask"]).reshape(batch_size, max_atoms)
        predictions["energy"].append(np.asarray(output["energy"][:valid_count]))
        predictions["forces"].append(
            np.asarray(forces).reshape(batch_size, max_atoms, 3)[:valid_count]
        )
        predictions["c6"].append(
            np.asarray(output["c6_coefficients"]).reshape(batch_size, max_atoms)[:valid_count]
        )
        predictions["alpha"].append(
            np.asarray(output["polarizabilities"]).reshape(batch_size, max_atoms)[:valid_count]
        )
        predictions["indices"].append(batch_indices[:valid_count])
        predictions["num_atoms"].append(atom_mask[:valid_count].sum(axis=1))

    return {
        "energy": np.concatenate(predictions["energy"], axis=0),
        "forces": _pad_atom_arrays(predictions["forces"]),
        "c6": _pad_atom_arrays(predictions["c6"]),
        "alpha": _pad_atom_arrays(predictions["alpha"]),
        "indices": np.concatenate(predictions["indices"], axis=0),
        "num_atoms": np.concatenate(predictions["num_atoms"], axis=0),
    }


def collect_targets(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    max_atoms: int,
) -> dict[str, np.ndarray]:
    return {
        "energy": cache["E_mbd"][indices].astype(np.float64),
        "forces": cache["F_mbd"][indices, :max_atoms].astype(np.float64),
        "c6": cache["C6_mbd"][indices, :max_atoms].astype(np.float64),
        "alpha": cache["alpha_mbd"][indices, :max_atoms].astype(np.float64),
        "atom_mask": cache["atom_mask"][indices, :max_atoms].astype(bool),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median_abs": float(np.median(np.abs(values))),
        "q95_abs": float(np.quantile(np.abs(values), 0.95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _component_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None,
    scale: float,
    unit: str,
) -> dict[str, Any]:
    if mask is not None:
        mask = np.broadcast_to(mask, target.shape)
        target = target[mask]
        prediction = prediction[mask]
    error = (prediction - target) * scale
    target_scaled = target * scale
    prediction_scaled = prediction * scale
    centered_target = target_scaled - np.mean(target_scaled)
    centered_prediction = prediction_scaled - np.mean(prediction_scaled)
    denominator = np.linalg.norm(centered_target) * np.linalg.norm(centered_prediction)
    correlation = float(np.dot(centered_target.ravel(), centered_prediction.ravel()) / denominator) if denominator else float("nan")
    return {
        "unit": unit,
        "target": {
            "mean": float(np.mean(target_scaled)),
            "std": float(np.std(target_scaled)),
            "min": float(np.min(target_scaled)),
            "max": float(np.max(target_scaled)),
        },
        "error": _summary(error),
        "bias": float(np.mean(error)),
        "correlation": correlation,
    }


def compute_metrics(
    target_energy: np.ndarray,
    prediction_energy: np.ndarray,
    target_forces: np.ndarray,
    prediction_forces: np.ndarray,
    target_c6: np.ndarray,
    prediction_c6: np.ndarray,
    target_alpha: np.ndarray,
    prediction_alpha: np.ndarray,
    atom_mask: np.ndarray,
    num_atoms: np.ndarray,
) -> dict[str, Any]:
    energy_error_hartree = prediction_energy - target_energy
    energy_error_kcal = energy_error_hartree * HARTREE_TO_KCAL_MOL
    force_error_au = prediction_forces - target_forces
    force_error_kcal = force_error_au * FORCE_AU_TO_KCAL_MOL_ANGSTROM
    per_structure_force_rmse = np.sqrt(
        np.sum(np.square(force_error_au) * atom_mask[:, :, None], axis=(1, 2))
        / np.maximum(3 * num_atoms, 1)
    )
    return {
        "num_structures": int(len(num_atoms)),
        "num_atoms": {
            "min": int(np.min(num_atoms)),
            "median": float(np.median(num_atoms)),
            "max": int(np.max(num_atoms)),
        },
        "energy_hartree": _component_metrics(
            target_energy,
            prediction_energy,
            None,
            1.0,
            "hartree",
        ),
        "energy_kcal_mol": _component_metrics(
            target_energy,
            prediction_energy,
            None,
            HARTREE_TO_KCAL_MOL,
            "kcal/mol",
        ),
        "energy_kcal_mol_per_atom": {
            "unit": "kcal/mol/atom",
            "error": _summary(energy_error_kcal / num_atoms),
            "bias": float(np.mean(energy_error_kcal / num_atoms)),
        },
        "forces_hartree_bohr": _component_metrics(
            target_forces,
            prediction_forces,
            atom_mask[:, :, None],
            1.0,
            "hartree/bohr",
        ),
        "forces_kcal_mol_angstrom": _component_metrics(
            target_forces,
            prediction_forces,
            atom_mask[:, :, None],
            FORCE_AU_TO_KCAL_MOL_ANGSTROM,
            "kcal/mol/angstrom",
        ),
        "forces_per_structure_rmse_hartree_bohr": {
            "unit": "hartree/bohr",
            "mean": float(np.mean(per_structure_force_rmse)),
            "median": float(np.median(per_structure_force_rmse)),
            "q95": float(np.quantile(per_structure_force_rmse, 0.95)),
            "max": float(np.max(per_structure_force_rmse)),
        },
        "c6_native": _component_metrics(
            target_c6,
            prediction_c6,
            atom_mask,
            1.0,
            "QCML native, assumed hartree*bohr^6",
        ),
        "c6_log1p": _component_metrics(
            np.log1p(target_c6),
            np.log1p(prediction_c6),
            atom_mask,
            1.0,
            "log1p(QCML native)",
        ),
        "polarizability_bohr3": _component_metrics(
            target_alpha,
            prediction_alpha,
            atom_mask,
            1.0,
            "bohr^3",
        ),
        "polarizability_log1p": _component_metrics(
            np.log1p(target_alpha),
            np.log1p(prediction_alpha),
            atom_mask,
            1.0,
            "log1p(bohr^3)",
        ),
    }


def _pad_atom_arrays(arrays: list[np.ndarray], pad_value: float | bool = 0) -> np.ndarray:
    """Pad atom-axis arrays from different shards to a common width."""
    max_atoms = max(array.shape[1] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[1] == max_atoms:
            padded.append(array)
            continue
        pad_spec = [(0, 0), (0, max_atoms - array.shape[1])]
        pad_spec.extend((0, 0) for _ in array.shape[2:])
        padded.append(np.pad(array, pad_spec, constant_values=pad_value))
    return np.concatenate(padded, axis=0)


def _identity_limits(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    if np.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = 0.04 * (high - low)
    return low - padding, high + padding


def plot_scatter(path: Path, target: np.ndarray, prediction: np.ndarray, title: str, unit: str) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    axis.scatter(target, prediction, s=7, alpha=0.35)
    limits = _identity_limits(target, prediction)
    axis.plot(limits, limits, color="black", linewidth=1, linestyle="--")
    axis.set(xlim=limits, ylim=limits, xlabel=f"Reference [{unit}]", ylabel=f"Prediction [{unit}]")
    axis.set_title(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_error_histogram(path: Path, errors: dict[str, tuple[np.ndarray, str]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for axis, (title, (values, unit)) in zip(axes.flat, errors.items(), strict=True):
        axis.hist(values, bins=60, alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel(f"Prediction - reference [{unit}]")
        axis.set_ylabel("Count")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_error_vs_atoms(
    path: Path,
    num_atoms: np.ndarray,
    energy_error_kcal: np.ndarray,
    force_rmse_kcal: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, values, ylabel in (
        (axes[0], energy_error_kcal, "Energy error [kcal/mol]"),
        (axes[1], force_rmse_kcal, "Force RMSE [kcal/mol/angstrom]"),
    ):
        axis.scatter(num_atoms, values, s=8, alpha=0.35)
        unique_atoms = np.unique(num_atoms)
        means = [np.mean(values[num_atoms == count]) for count in unique_atoms]
        axis.plot(unique_atoms, means, color="black", linewidth=1.5, label="mean")
        axis.set_xlabel("Number of atoms")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_per_structure_csv(
    path: Path,
    shard_ids: list[str],
    dataset_indices: np.ndarray,
    num_atoms: np.ndarray,
    target_energy: np.ndarray,
    prediction_energy: np.ndarray,
    target_forces: np.ndarray,
    prediction_forces: np.ndarray,
    atom_mask: np.ndarray,
) -> None:
    fieldnames = [
        "shard",
        "dataset_index",
        "num_atoms",
        "target_energy_hartree",
        "prediction_energy_hartree",
        "energy_error_hartree",
        "energy_error_kcal_mol",
        "energy_error_kcal_mol_per_atom",
        "force_rmse_hartree_bohr",
        "force_rmse_kcal_mol_angstrom",
        "force_max_abs_hartree_bohr",
    ]
    force_error = prediction_forces - target_forces
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in range(len(num_atoms)):
            masked_error = force_error[row][atom_mask[row]]
            force_rmse = float(np.sqrt(np.mean(np.square(masked_error)))) if len(masked_error) else 0.0
            energy_error = float(prediction_energy[row] - target_energy[row])
            writer.writerow(
                {
                    "shard": shard_ids[row],
                    "dataset_index": int(dataset_indices[row]),
                    "num_atoms": int(num_atoms[row]),
                    "target_energy_hartree": float(target_energy[row]),
                    "prediction_energy_hartree": float(prediction_energy[row]),
                    "energy_error_hartree": energy_error,
                    "energy_error_kcal_mol": energy_error * HARTREE_TO_KCAL_MOL,
                    "energy_error_kcal_mol_per_atom": energy_error * HARTREE_TO_KCAL_MOL / num_atoms[row],
                    "force_rmse_hartree_bohr": force_rmse,
                    "force_rmse_kcal_mol_angstrom": force_rmse * FORCE_AU_TO_KCAL_MOL_ANGSTROM,
                    "force_max_abs_hartree_bohr": float(np.max(np.abs(masked_error))) if len(masked_error) else 0.0,
                }
            )


def write_report(
    output_dir: Path,
    shard_ids: list[str],
    dataset_indices: np.ndarray,
    target: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    num_atoms: np.ndarray,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atom_mask = target["atom_mask"]
    metrics = compute_metrics(
        target["energy"],
        prediction["energy"],
        target["forces"],
        prediction["forces"],
        target["c6"],
        prediction["c6"],
        target["alpha"],
        prediction["alpha"],
        atom_mask,
        num_atoms,
    )
    metrics["units"] = {
        "energy_native": "hartree",
        "energy_regular": "kcal/mol",
        "forces_native": "hartree/bohr",
        "forces_regular": "kcal/mol/angstrom",
        "c6": "QCML native, assumed hartree*bohr^6",
        "polarizability": "bohr^3",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_per_structure_csv(
        output_dir / "per_structure_errors.csv",
        shard_ids,
        dataset_indices,
        num_atoms,
        target["energy"],
        prediction["energy"],
        target["forces"],
        prediction["forces"],
        atom_mask,
    )

    valid_forces = np.broadcast_to(atom_mask[:, :, None], target["forces"].shape)
    valid_atoms = atom_mask
    plot_scatter(
        output_dir / "scatter_energy_kcal_mol.png",
        target["energy"] * HARTREE_TO_KCAL_MOL,
        prediction["energy"] * HARTREE_TO_KCAL_MOL,
        "MBD energy",
        "kcal/mol",
    )
    plot_scatter(
        output_dir / "scatter_forces_kcal_mol_angstrom.png",
        target["forces"][valid_forces] * FORCE_AU_TO_KCAL_MOL_ANGSTROM,
        prediction["forces"][valid_forces] * FORCE_AU_TO_KCAL_MOL_ANGSTROM,
        "MBD force components",
        "kcal/mol/angstrom",
    )
    plot_scatter(
        output_dir / "scatter_c6_native.png",
        target["c6"][valid_atoms],
        prediction["c6"][valid_atoms],
        "Atomic C6",
        "QCML native",
    )
    plot_scatter(
        output_dir / "scatter_polarizability_bohr3.png",
        target["alpha"][valid_atoms],
        prediction["alpha"][valid_atoms],
        "Atomic polarizability",
        "bohr^3",
    )

    force_error = prediction["forces"] - target["forces"]
    per_structure_force_rmse = np.sqrt(
        np.sum(np.square(force_error) * atom_mask[:, :, None], axis=(1, 2))
        / np.maximum(3 * num_atoms, 1)
    )
    plot_error_histogram(
        output_dir / "error_distributions.png",
        {
            "Energy": (
                (prediction["energy"] - target["energy"]) * HARTREE_TO_KCAL_MOL,
                "kcal/mol",
            ),
            "Forces": (
                force_error[valid_forces] * FORCE_AU_TO_KCAL_MOL_ANGSTROM,
                "kcal/mol/angstrom",
            ),
            "C6": (prediction["c6"][valid_atoms] - target["c6"][valid_atoms], "QCML native"),
            "Polarizability": (
                prediction["alpha"][valid_atoms] - target["alpha"][valid_atoms],
                "bohr^3",
            ),
        },
    )
    plot_error_vs_atoms(
        output_dir / "error_vs_num_atoms.png",
        num_atoms,
        (prediction["energy"] - target["energy"]) * HARTREE_TO_KCAL_MOL,
        per_structure_force_rmse * FORCE_AU_TO_KCAL_MOL_ANGSTROM,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bucket-width", type=int, default=8)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--data-split", type=Path)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument("--test-shards", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--max-atoms", type=int)
    args = parser.parse_args()

    model, params = load_model(args.checkpoint)
    shard_paths = resolve_split_paths(
        args.cache,
        args.checkpoint,
        args.split,
        args.validation_shards,
        args.test_shards,
        args.data_split,
    )
    all_targets: dict[str, list[np.ndarray]] = {
        "energy": [],
        "forces": [],
        "c6": [],
        "alpha": [],
        "atom_mask": [],
    }
    all_predictions: dict[str, list[np.ndarray]] = {
        "energy": [],
        "forces": [],
        "c6": [],
        "alpha": [],
    }
    all_indices = []
    all_num_atoms = []
    all_shard_ids: list[str] = []
    remaining = args.max_structures

    if shard_paths is None:
        cache = restore_cache(args.cache)
        indices = single_shard_indices(
            cache,
            args.split,
            args.validation_fraction,
            args.seed,
            args.max_atoms,
        )
        if remaining is not None:
            indices = indices[:remaining]
        shard_paths = [args.cache]
        shard_indices = [(args.cache, cache, indices)]
    else:
        shard_indices = []
        for shard_path in shard_paths:
            if remaining is not None and remaining <= 0:
                break
            cache = restore_cache(shard_path)
            indices = eligible_indices(cache, args.max_atoms)
            if remaining is not None:
                indices = indices[:remaining]
                remaining -= len(indices)
            shard_indices.append((shard_path, cache, indices))

    for shard_path, cache, indices in shard_indices:
        if not len(indices):
            continue
        prediction = predict_shard(
            model,
            params,
            cache,
            indices,
            args.batch_size,
            args.bucket_width,
        )
        max_atoms = prediction["forces"].shape[1]
        targets = collect_targets(cache, prediction["indices"], max_atoms)
        num_atoms = prediction["num_atoms"].astype(np.int64)
        for key in all_targets:
            all_targets[key].append(targets[key])
        for key in all_predictions:
            all_predictions[key].append(prediction[key])
        all_indices.append(prediction["indices"])
        all_num_atoms.append(num_atoms)
        all_shard_ids.extend([shard_path.name] * len(num_atoms))

    if not all_indices:
        raise ValueError(f"The selected {args.split} split contains no eligible structures")

    target = {
        "energy": np.concatenate(all_targets["energy"], axis=0),
        "forces": _pad_atom_arrays(all_targets["forces"]),
        "c6": _pad_atom_arrays(all_targets["c6"]),
        "alpha": _pad_atom_arrays(all_targets["alpha"]),
        "atom_mask": _pad_atom_arrays(all_targets["atom_mask"], pad_value=False),
    }
    prediction = {
        "energy": np.concatenate(all_predictions["energy"], axis=0),
        "forces": _pad_atom_arrays(all_predictions["forces"]),
        "c6": _pad_atom_arrays(all_predictions["c6"]),
        "alpha": _pad_atom_arrays(all_predictions["alpha"]),
    }
    dataset_indices = np.concatenate(all_indices, axis=0)
    num_atoms = np.concatenate(all_num_atoms, axis=0)
    metrics = write_report(
        args.output_dir,
        all_shard_ids,
        dataset_indices,
        target,
        prediction,
        num_atoms,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote MBD report to {args.output_dir}")


if __name__ == "__main__":
    main()
