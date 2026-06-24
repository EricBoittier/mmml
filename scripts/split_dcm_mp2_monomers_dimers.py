#!/usr/bin/env python3
"""Split padded DCM MP2 NPZ data into PSF-ordered monomer and dimer files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_permutation(text: str) -> np.ndarray:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if sorted(values) != list(range(len(values))):
        raise ValueError("permutation must be zero-based, e.g. 0,3,4,1,2")
    return np.asarray(values, dtype=int)


def _repeat_perm(perm: np.ndarray, active_n: int) -> np.ndarray:
    if active_n % len(perm) != 0:
        raise ValueError(f"N={active_n} is not divisible by monomer size {len(perm)}")
    return np.concatenate([perm + offset for offset in range(0, active_n, len(perm))])


def split_dcm_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    monomer_permutation: np.ndarray,
) -> dict[str, Path]:
    with np.load(input_path, allow_pickle=True) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}

    required = {"N", "E", "Z", "R", "F"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"{input_path} is missing required keys: {missing}")

    counts = np.asarray(arrays["N"], dtype=int)
    active_values = sorted(set(counts.tolist()))
    for active_n in active_values:
        mask = counts == active_n
        perm = _repeat_perm(monomer_permutation, active_n)
        for key in ("Z", "R", "F"):
            arrays[key][mask, :active_n] = arrays[key][mask][:, perm]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    outputs: dict[str, Path] = {}
    for label, active_n in (("monomers", len(monomer_permutation)), ("dimers", 2 * len(monomer_permutation))):
        mask = counts == active_n
        indices = np.where(mask)[0]
        payload = {key: value[mask] for key, value in arrays.items()}
        payload["source_indices"] = indices.astype(np.int64)
        path = output_dir / f"{stem}_{label}_psf_order.npz"
        np.savez_compressed(path, **payload)
        outputs[label] = path

    metadata = {
        "input": str(input_path),
        "monomer_permutation": monomer_permutation.tolist(),
        "count_histogram": {str(int(n)): int(np.sum(counts == n)) for n in active_values},
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    metadata_path = output_dir / f"{stem}_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    outputs["metadata"] = metadata_path
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--monomer-permutation",
        default="0,3,4,1,2",
        help="Zero-based monomer reorder; default maps C,H,H,Cl,Cl -> C,Cl,Cl,H,H.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = split_dcm_dataset(
        args.input,
        args.output_dir,
        monomer_permutation=_parse_permutation(args.monomer_permutation),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
