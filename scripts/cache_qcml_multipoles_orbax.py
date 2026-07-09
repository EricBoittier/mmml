#!/usr/bin/env python3
"""Cache QCML molecular inputs and traceless multipoles in Orbax."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

from mmml.models.multipoles.representations import irrep_blocks_to_traceless


DEFAULT_DATASET = "qcml/dft_multipole_moments"
DEFAULT_CACHE_DIR = Path("orbax_cache/qcml_multipoles_traceless")
FIELD_ALIASES = {
    "R": ("R", "positions", "coordinates"),
    "Z": ("Z", "atomic_numbers", "nuclear_charges"),
    "Q": ("Q", "charge", "total_charge"),
    "S": ("S", "spin", "multiplicity"),
    "multipoles": ("multipole_moments", "multipoles"),
    "key_hash": ("key_hash",),
}


def _field(example: dict[str, Any], logical_name: str, *, required: bool = True) -> Any:
    for name in FIELD_ALIASES[logical_name]:
        if name in example:
            return example[name]
    if required:
        aliases = ", ".join(FIELD_ALIASES[logical_name])
        raise KeyError(f"Missing {logical_name}; expected one of: {aliases}")
    return None


def preprocess_examples(
    examples: Any,
    *,
    max_degree: int = 3,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert a stream of NumPy examples to a stackable Orbax PyTree."""
    records: dict[str, list[np.ndarray]] = {
        "R": [],
        "Z": [],
        "Q": [],
        "S": [],
        "atom_mask": [],
        "multipoles": [],
    }
    key_hashes: list[np.ndarray] = []
    converted_examples = []

    for index, example in enumerate(examples):
        if limit is not None and index >= limit:
            break
        positions = np.asarray(_field(example, "R"))
        converted = irrep_blocks_to_traceless(
            np.asarray(_field(example, "multipoles")),
            max_degree=max_degree,
        )
        converted_examples.append(
            {
                "R": positions,
                "Z": np.asarray(_field(example, "Z")),
                "Q": np.asarray(_field(example, "Q")),
                "S": np.asarray(_field(example, "S")),
                "multipoles": np.asarray(_field(example, "multipoles")),
                **{key: np.asarray(value) for key, value in converted.items()},
            }
        )
        key_hash = _field(example, "key_hash", required=False)
        if key_hash is not None:
            key_hashes.append(np.asarray(key_hash))
        if index % 1000 == 0:
            print(f"Processed {index} examples")

    if not converted_examples:
        raise ValueError("Dataset produced no examples")

    max_atoms = max(record["R"].shape[0] for record in converted_examples)
    tensor_keys = [
        key
        for key in converted_examples[0]
        if key not in {"R", "Z", "Q", "S", "multipoles"}
    ]
    for key in tensor_keys:
        records[key] = []

    for record in converted_examples:
        num_atoms = record["R"].shape[0]
        records["R"].append(np.pad(record["R"], ((0, max_atoms - num_atoms), (0, 0))))
        records["Z"].append(np.pad(record["Z"], (0, max_atoms - num_atoms)))
        records["atom_mask"].append(
            np.pad(np.ones(num_atoms, dtype=np.float32), (0, max_atoms - num_atoms))
        )
        for key in ("Q", "S", "multipoles", *tensor_keys):
            records[key].append(record[key])

    cache = {key: np.stack(values) for key, values in records.items()}
    if len(key_hashes) == len(converted_examples):
        cache["key_hash"] = np.stack(key_hashes)
    return cache


def save_orbax_cache(cache: dict[str, np.ndarray], checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(checkpoint, cache)


def load_orbax_cache(checkpoint: Path) -> dict[str, np.ndarray]:
    return ocp.PyTreeCheckpointer().restore(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "tensorflow-datasets is required for this command; install the QCML "
            "dataset environment first"
        ) from exc

    dataset = tfds.load(args.dataset, split=args.split, data_dir=args.data_dir)
    cache = preprocess_examples(
        tfds.as_numpy(dataset),
        max_degree=args.max_degree,
        limit=args.limit,
    )
    checkpoint = args.cache_dir / "0"
    save_orbax_cache(cache, checkpoint)
    restored = load_orbax_cache(checkpoint)
    print(f"Saved {restored['R'].shape[0]} examples to {checkpoint}")
    print(f"Packed multipole shape: {restored['multipoles'].shape}")


if __name__ == "__main__":
    main()
