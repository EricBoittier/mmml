#!/usr/bin/env python
"""Pad heterogeneous PhysNet NPZs to a shared atom axis and concatenate.

DCM dimers are 10 atoms, ACO dimers 20, ACO–DCM heteros 15. Training via
``prepare_multiple_datasets`` requires a single padded width. Padding atoms get
Z/F = 0 and ``mol_id`` / ``cgenff_type_idx`` = -1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

# Keys that are (n_frames, n_atoms[, 3]) and need atom-axis padding.
_ATOM_KEYS = ("R", "Z", "F", "mol_id", "cgenff_type_idx", "cgenff_charge")
# Keys that are per-frame only (concat on axis 0).
_FRAME_KEYS = ("N", "E", "D", "E_total", "Q", "res_name")


def pad_frames_to(
    data: dict[str, Any],
    pad_to: int,
) -> dict[str, np.ndarray]:
    """Return a copy of *data* with atom-axis arrays padded to *pad_to*."""
    n = int(np.asarray(data["N"]).shape[0])
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        a = np.asarray(arr)
        if key in _ATOM_KEYS and a.ndim >= 1 and a.shape[0] == n:
            if a.ndim == 1:
                # unexpected flat; skip
                out[key] = a
                continue
            cur = a.shape[1]
            if cur > pad_to:
                raise ValueError(
                    f"{key}: atom axis {cur} exceeds pad_to={pad_to}"
                )
            if cur == pad_to:
                out[key] = a
                continue
            if key in ("mol_id", "cgenff_type_idx"):
                fill = -1
                dtype = np.int32
            elif key == "Z":
                fill = 0
                dtype = np.int32
            else:
                fill = 0.0
                dtype = a.dtype
            shape = (n, pad_to) + a.shape[2:]
            padded = np.full(shape, fill, dtype=dtype)
            padded[:, :cur, ...] = a
            out[key] = padded
        else:
            out[key] = a
    return out


_REQUIRED_KEYS = ("R", "Z", "N")


def merge_npz_paths(
    paths: list[Path],
    pad_to: int | None = None,
) -> dict[str, np.ndarray]:
    """Load one or more NPZs, pad to a shared width, concatenate on axis 0."""
    loaded = [dict(np.load(p, allow_pickle=True)) for p in paths]
    if not loaded:
        raise ValueError("no NPZ paths given")
    for path, data in zip(paths, loaded):
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"{path}: missing required key(s) {missing} (need R/Z/N)"
            )
    widths = [int(np.asarray(d["R"]).shape[1]) for d in loaded]
    target = int(pad_to) if pad_to is not None else max(widths)
    padded = [pad_frames_to(d, target) for d in loaded]

    merged: dict[str, np.ndarray] = {}
    # Prefer keys present in the first file; R/Z/N already validated above.
    keys = list(padded[0].keys())
    for key in keys:
        chunks = [d[key] for d in padded if key in d]
        if not chunks:
            continue
        if key in ("_mmml_units", "atom_ref_energies",
                   "cgenff_master_sigmas", "cgenff_master_epsilons"):
            # Keep from the first file that has them (LoT / master tables).
            merged[key] = chunks[0]
            continue
        try:
            merged[key] = np.concatenate(chunks, axis=0)
        except ValueError as exc:
            raise ValueError(
                f"cannot concatenate key {key!r}: {[c.shape for c in chunks]}"
            ) from exc
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path, help="input NPZ path(s)")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument(
        "--pad-to",
        type=int,
        default=None,
        help="atom-axis width (default: max width among inputs)",
    )
    args = p.parse_args(argv)
    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"ERROR: missing input {path}")
    try:
        merged = merge_npz_paths(args.inputs, pad_to=args.pad_to)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    if "N" not in merged or "R" not in merged:
        raise SystemExit("ERROR: merged NPZ missing required key(s) R and/or N")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **merged)
    n = len(merged["N"])
    w = int(merged["R"].shape[1])
    print(f"pad-merge: {n} frames, pad={w} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
