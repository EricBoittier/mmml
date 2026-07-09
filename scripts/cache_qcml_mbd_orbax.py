#!/usr/bin/env python3
"""Join QCML MBD partitions and cache padded arrays with Orbax."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from mmml.data.orbax_shards import write_orbax_shards

DATASETS = {
    "geometry": "qcml/dft_force_field",
    "c6": "qcml/dft_mbd_c6_coefficients",
    "correction": "qcml/dft_mbd_correction",
    "polarizabilities": "qcml/dft_mbd_polarizabilities",
}


def _hash(example: dict[str, Any]) -> bytes:
    return bytes(np.asarray(example["key_hash"]).item())


def preprocess_examples(
    examples: Any,
    *,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert matched geometry/C6/correction/polarizability records."""
    records = []
    hashes = []
    for index, (geometry, c6, correction, polarizabilities) in enumerate(examples):
        if limit is not None and index >= limit:
            break
        current_hashes = tuple(
            _hash(example)
            for example in (geometry, c6, correction, polarizabilities)
        )
        if len(set(current_hashes)) != 1:
            raise ValueError(f"Dataset key mismatch at example {index}: {current_hashes!r}")

        positions = np.asarray(geometry["positions"], dtype=np.float32)
        atomic_numbers = np.asarray(geometry["atomic_numbers"], dtype=np.int32)
        c6_values = np.asarray(c6["mbd_c6_coefficients"], dtype=np.float32)
        alpha_values = np.asarray(
            polarizabilities["mbd_polarizabilities"],
            dtype=np.float32,
        )
        forces = np.asarray(correction["mbd_forces"], dtype=np.float32)
        num_atoms = len(atomic_numbers)
        for name, value, expected in (
            ("positions", positions, (num_atoms, 3)),
            ("mbd_forces", forces, (num_atoms, 3)),
            ("mbd_c6_coefficients", c6_values, (num_atoms,)),
            ("mbd_polarizabilities", alpha_values, (num_atoms,)),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        records.append(
            {
                "R": positions,
                "Z": atomic_numbers,
                "Q": np.asarray(geometry["charge"], dtype=np.float32),
                "S": np.asarray(geometry["multiplicity"], dtype=np.float32),
                "E_mbd": np.asarray(correction["mbd_energy"], dtype=np.float64),
                "F_mbd": forces,
                "C6_mbd": c6_values,
                "alpha_mbd": alpha_values,
            }
        )
        hashes.append(current_hashes[0])
        if index % 1000 == 0:
            print(f"Processed {index} examples")

    if not records:
        raise ValueError("Dataset produced no examples")
    max_atoms = max(len(record["Z"]) for record in records)
    cache: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("R", "Z", "Q", "S", "E_mbd", "F_mbd", "C6_mbd", "alpha_mbd", "atom_mask")
    }
    for record in records:
        num_atoms = len(record["Z"])
        padding = max_atoms - num_atoms
        cache["R"].append(np.pad(record["R"], ((0, padding), (0, 0))))
        cache["Z"].append(np.pad(record["Z"], (0, padding)))
        cache["F_mbd"].append(np.pad(record["F_mbd"], ((0, padding), (0, 0))))
        cache["C6_mbd"].append(np.pad(record["C6_mbd"], (0, padding)))
        cache["alpha_mbd"].append(np.pad(record["alpha_mbd"], (0, padding)))
        cache["atom_mask"].append(
            np.pad(np.ones(num_atoms, dtype=np.float32), (0, padding))
        )
        for key in ("Q", "S", "E_mbd"):
            cache[key].append(record[key])

    result = {key: np.stack(values) for key, values in cache.items()}
    max_hash_length = max(map(len, hashes))
    result["key_hash"] = np.stack(
        [
            np.pad(np.frombuffer(value, dtype=np.uint8), (0, max_hash_length - len(value)))
            for value in hashes
        ]
    )
    result["key_hash_length"] = np.asarray([len(value) for value in hashes])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("orbax_cache/qcml_mbd"))
    parser.add_argument("--split", default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=50000)
    args = parser.parse_args()

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit("tensorflow-datasets is required for this command") from exc

    read_config = tfds.ReadConfig(interleave_cycle_length=1)
    datasets = [
        tfds.load(
            dataset,
            split=args.split,
            data_dir=args.data_dir,
            read_config=read_config,
        )
        for dataset in DATASETS.values()
    ]
    examples = zip(*(tfds.as_numpy(dataset) for dataset in datasets))
    if args.limit is not None:
        examples = itertools.islice(examples, args.limit)
    manifest = write_orbax_shards(
        examples,
        args.cache_dir,
        preprocess_examples,
        shard_size=args.shard_size,
        dataset_kind="qcml_mbd",
    )
    print(f"Saved sharded MBD cache: {manifest}")


if __name__ == "__main__":
    main()
