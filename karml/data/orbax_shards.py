"""Streaming helpers for manifest-based Orbax dataset shards."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


MANIFEST_NAME = "manifest.json"


def chunked(examples: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("shard_size must be positive")
    chunk = []
    for example in examples:
        chunk.append(example)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_orbax_shards(
    examples: Iterable[Any],
    root: Path,
    preprocess: Callable[[Iterable[Any]], dict[str, Any]],
    *,
    shard_size: int,
    dataset_kind: str,
) -> Path:
    """Preprocess and save one bounded chunk at a time."""
    root.mkdir(parents=True, exist_ok=True)
    shards = []
    total = 0
    for shard_index, records in enumerate(chunked(examples, shard_size)):
        payload = preprocess(records)
        count = int(payload["R"].shape[0])
        shard_name = f"shard-{shard_index:05d}"
        ocp.PyTreeCheckpointer().save(root / shard_name, payload)
        shards.append(
            {
                "path": shard_name,
                "num_structures": count,
                "max_atoms": int(payload["R"].shape[1]),
            }
        )
        total += count
        print(f"Saved {shard_name}: {count} structures")
    if not shards:
        raise ValueError("Dataset produced no examples")
    manifest = {
        "format": "mmml-orbax-shards-v1",
        "dataset_kind": dataset_kind,
        "num_structures": total,
        "shard_size": shard_size,
        "shards": shards,
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))


def iter_restored_shards(root: Path) -> Iterator[dict[str, Any]]:
    for shard in read_manifest(root)["shards"]:
        yield ocp.PyTreeCheckpointer().restore(root / shard["path"])


def partition_shards(
    root: Path,
    *,
    validation_shards: int,
    test_shards: int,
) -> dict[str, list[Path]]:
    """Partition ordered shards, reserving validation/test shards at the end."""
    if validation_shards < 0 or test_shards < 0:
        raise ValueError("validation_shards and test_shards must be non-negative")
    paths = [root / shard["path"] for shard in read_manifest(root)["shards"]]
    reserved = validation_shards + test_shards
    if reserved >= len(paths):
        raise ValueError(
            f"Need at least one training shard; got {len(paths)} total and "
            f"{reserved} reserved"
        )
    test_start = len(paths) - test_shards if test_shards else len(paths)
    validation_start = test_start - validation_shards
    return {
        "train": paths[:validation_start],
        "validation": paths[validation_start:test_start],
        "test": paths[test_start:],
    }
