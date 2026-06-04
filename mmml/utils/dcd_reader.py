"""Minimal CHARMM DCD trajectory reader (coordinates only)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def read_dcd_trajectory(
    path: PathLike,
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Read all frames from a CHARMM/NAMD DCD file.

    Returns
    -------
    positions
        float64 array of shape ``(n_frames, n_atoms, 3)``.
    header
        Dict with ``n_frames``, ``n_atoms``, ``nsavc``, ``dt``.
    """
    path = Path(path)
    with path.open("rb") as f:
        rec1 = struct.unpack("<i", f.read(4))[0]
        cord = f.read(4)
        if rec1 != 84 or cord != b"CORD":
            raise ValueError(f"Not a CHARMM DCD file: {path}")
        n_frames = struct.unpack("<i", f.read(4))[0]
        f.read(4)  # istart
        nsavc = struct.unpack("<i", f.read(4))[0]
        f.read(4 * 6)
        dt = struct.unpack("<f", f.read(4))[0]
        iscell = struct.unpack("<i", f.read(4))[0]
        f.read(4 * 9)
        struct.unpack("<i", f.read(4))[0]
        block_size = struct.unpack("<i", f.read(4))[0]
        _n_titles = struct.unpack("<i", f.read(4))[0]
        f.read(block_size - 4)
        f.read(4)
        struct.unpack("<i", f.read(4))[0]
        n_atoms = struct.unpack("<i", f.read(4))[0]
        struct.unpack("<i", f.read(4))[0]

        has_unitcell = iscell != 0
        frames: list[np.ndarray] = []
        n_read = 0
        while n_read < n_frames:
            if max_frames is not None and n_read >= max_frames:
                break
            if has_unitcell:
                rec = struct.unpack("<i", f.read(4))[0]
                if rec == 0:
                    break
                f.read(rec)
                struct.unpack("<i", f.read(4))[0]
            rec = f.read(4)
            if len(rec) < 4:
                raise ValueError(
                    f"Truncated DCD at frame {n_read + 1}/{n_frames}: {path} "
                    f"(file size {path.stat().st_size} bytes; dynamics may have aborted mid-write)"
                )
            struct.unpack("<i", rec)[0]
            x = np.fromfile(f, dtype=np.float32, count=n_atoms)
            struct.unpack("<i", f.read(4))[0]
            struct.unpack("<i", f.read(4))[0]
            y = np.fromfile(f, dtype=np.float32, count=n_atoms)
            struct.unpack("<i", f.read(4))[0]
            struct.unpack("<i", f.read(4))[0]
            z = np.fromfile(f, dtype=np.float32, count=n_atoms)
            struct.unpack("<i", f.read(4))[0]
            if x.size != n_atoms:
                break
            frames.append(np.stack([x, y, z], axis=1).astype(np.float64))
            n_read += 1

    if not frames:
        raise ValueError(f"No frames read from DCD: {path}")
    pos = np.stack(frames, axis=0)
    return pos, {
        "n_frames": len(frames),
        "n_atoms": n_atoms,
        "nsavc": nsavc,
        "dt": float(dt),
    }
