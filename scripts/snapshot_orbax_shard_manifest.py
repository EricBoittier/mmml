#!/usr/bin/env python3
"""Create a stable manifest snapshot from completed Orbax shard directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import orbax.checkpoint as ocp


def is_complete_orbax_shard(path: Path) -> bool:
    return (
        (path / "_CHECKPOINT_METADATA").exists()
        or (path / "manifest.ocdbt").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset-kind", required=True)
    parser.add_argument(
        "--exclude-newest",
        type=int,
        default=1,
        help="Exclude this many newest complete shards in case a cache job is active.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.exclude_newest < 0:
        raise ValueError("exclude_newest must be non-negative")

    paths = sorted(
        path
        for path in args.cache.glob("shard-*")
        if path.is_dir() and is_complete_orbax_shard(path)
    )
    if args.exclude_newest:
        paths = paths[: -args.exclude_newest]
    if not paths:
        raise ValueError("No stable completed shards found")

    shards = []
    total = 0
    for path in paths:
        restored = ocp.PyTreeCheckpointer().restore(path)
        count, max_atoms = restored["R"].shape[:2]
        shards.append(
            {
                "path": path.name,
                "num_structures": int(count),
                "max_atoms": int(max_atoms),
            }
        )
        total += int(count)
        del restored

    manifest = {
        "format": "mmml-orbax-shards-v1",
        "dataset_kind": args.dataset_kind,
        "num_structures": total,
        "shard_size": max(item["num_structures"] for item in shards),
        "snapshot_excluded_newest": args.exclude_newest,
        "shards": shards,
    }
    output = args.output or args.cache / "manifest.snapshot.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output}: {len(shards)} shards, {total} structures")


if __name__ == "__main__":
    main()
