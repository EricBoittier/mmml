#!/usr/bin/env python3
"""Cache QCML molecular inputs and traceless multipoles in Orbax."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

from mmml.models.multipoles.representations import irrep_blocks_to_traceless
from mmml.data.orbax_shards import write_orbax_shards


DEFAULT_DATASET = "qcml/dft_multipole_moments"
DEFAULT_GEOMETRY_DATASET = "qcml/dft_force_field"
DEFAULT_CACHE_DIR = Path("orbax_cache/qcml_multipoles_traceless")
FIELD_ALIASES = {
    "R": ("R", "positions", "coordinates"),
    "Z": ("Z", "atomic_numbers", "nuclear_charges"),
    "Q": ("Q", "charge", "total_charge"),
    "S": ("S", "multiplicity", "spin"),
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
    """Convert matched ``(force_field, multipoles)`` examples to an Orbax PyTree."""
    records: dict[str, list[np.ndarray]] = {
        "R": [],
        "Z": [],
        "Q": [],
        "S": [],
        "atom_mask": [],
        "multipoles": [],
    }
    key_hashes: list[bytes] = []
    converted_examples = []

    for index, pair in enumerate(examples):
        if limit is not None and index >= limit:
            break
        geometry, moments = pair
        geometry_hash = bytes(np.asarray(_field(geometry, "key_hash")).item())
        moments_hash = bytes(np.asarray(_field(moments, "key_hash")).item())
        if geometry_hash != moments_hash:
            raise ValueError(
                f"Dataset key mismatch at example {index}: "
                f"{geometry_hash!r} != {moments_hash!r}"
            )

        positions = np.asarray(_field(geometry, "R"))
        charge = np.asarray(_field(geometry, "Q"))
        blocks = {"l0_irrep": np.atleast_1d(charge).astype(positions.dtype)}
        source_names = {
            1: "pbe0_dipole",
            2: "pbe0_quadrupole",
            3: "pbe0_octupole",
        }
        for degree in range(1, max_degree + 1):
            source_name = source_names.get(degree)
            if source_name is None or source_name not in moments:
                raise KeyError(f"Missing spherical target for degree l={degree}")
            block = np.asarray(moments[source_name])
            expected_width = 2 * degree + 1
            if block.shape != (expected_width,):
                raise ValueError(
                    f"{source_name} must have shape ({expected_width},), got {block.shape}"
                )
            blocks[f"l{degree}_irrep"] = block
        packed = np.concatenate([blocks[f"l{degree}_irrep"] for degree in range(max_degree + 1)])
        converted = irrep_blocks_to_traceless(packed, max_degree=max_degree)
        converted_examples.append(
            {
                "R": positions,
                "Z": np.asarray(_field(geometry, "Z")),
                "Q": charge,
                "S": np.asarray(_field(geometry, "S")),
                "multipoles": packed,
                **{key: np.asarray(value) for key, value in converted.items()},
            }
        )
        key_hashes.append(geometry_hash)
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
    max_hash_length = max(map(len, key_hashes))
    cache["key_hash"] = np.stack(
        [
            np.pad(np.frombuffer(key_hash, dtype=np.uint8), (0, max_hash_length - len(key_hash)))
            for key_hash in key_hashes
        ]
    )
    cache["key_hash_length"] = np.asarray([len(key_hash) for key_hash in key_hashes])
    return cache


def save_orbax_cache(cache: dict[str, np.ndarray], checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(checkpoint, cache)


def load_orbax_cache(checkpoint: Path) -> dict[str, np.ndarray]:
    return ocp.PyTreeCheckpointer().restore(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--geometry-dataset", default=DEFAULT_GEOMETRY_DATASET)
    parser.add_argument("--split", default="full")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "tensorflow-datasets is required for this command; install the QCML "
            "dataset environment first"
        ) from exc

    read_config = tfds.ReadConfig(interleave_cycle_length=1)
    geometry_dataset = tfds.load(
        args.geometry_dataset,
        split=args.split,
        data_dir=args.data_dir,
        read_config=read_config,
    )
    multipole_dataset = tfds.load(
        args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        read_config=read_config,
    )
    examples = zip(tfds.as_numpy(geometry_dataset), tfds.as_numpy(multipole_dataset))
    if args.limit is not None:
        examples = itertools.islice(examples, args.limit)
    manifest = write_orbax_shards(
        examples,
        args.cache_dir,
        lambda records: preprocess_examples(records, max_degree=args.max_degree),
        shard_size=args.shard_size,
        dataset_kind="qcml_multipoles",
    )
    print(f"Saved sharded multipole cache: {manifest}")


if __name__ == "__main__":
    main()
