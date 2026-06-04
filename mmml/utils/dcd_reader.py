"""Minimal CHARMM DCD trajectory reader (coordinates only)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def _read_dcd_header(f) -> tuple[int, int, int, float, bool]:
    """Return ``n_frames``, ``n_atoms``, ``nsavc``, ``dt``, ``has_unitcell``."""
    rec1 = struct.unpack("<i", f.read(4))[0]
    cord = f.read(4)
    if rec1 != 84 or cord != b"CORD":
        raise ValueError("Not a CHARMM DCD file")
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
    return int(n_frames), int(n_atoms), int(nsavc), float(dt), iscell != 0


def scan_dcd_frame_count(path: PathLike) -> tuple[int, int, bool]:
    """Count readable coordinate frames (handles truncated files).

    Returns
    -------
    readable
        Frames successfully read from coordinate records.
    header_frames
        Frame count from the DCD header (may exceed ``readable`` if truncated).
    truncated
        True if the file ended before ``header_frames`` frames were read.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 16:
        return 0, 0, False
    with path.open("rb") as f:
        try:
            n_frames, n_atoms, _, _, has_unitcell = _read_dcd_header(f)
        except (struct.error, ValueError):
            return 0, 0, False
        if n_frames <= 0 or n_atoms <= 0:
            return 0, n_frames, False

        n_read = 0
        truncated = False
        while n_read < n_frames:
            if has_unitcell:
                rec_b = f.read(4)
                if len(rec_b) < 4:
                    truncated = True
                    break
                rec = struct.unpack("<i", rec_b)[0]
                if rec == 0:
                    break
                if len(f.read(rec)) < rec:
                    truncated = True
                    break
                if len(f.read(4)) < 4:
                    truncated = True
                    break
            rec_b = f.read(4)
            if len(rec_b) < 4:
                truncated = True
                break
            struct.unpack("<i", rec_b)[0]
            x = np.fromfile(f, dtype=np.float32, count=n_atoms)
            if len(f.read(4)) < 4:
                truncated = True
                break
            if len(f.read(4)) < 4:
                truncated = True
                break
            y = np.fromfile(f, dtype=np.float32, count=n_atoms)
            if len(f.read(4)) < 4:
                truncated = True
                break
            if len(f.read(4)) < 4:
                truncated = True
                break
            z = np.fromfile(f, dtype=np.float32, count=n_atoms)
            if len(f.read(4)) < 4:
                truncated = True
                break
            if x.size != n_atoms or y.size != n_atoms or z.size != n_atoms:
                truncated = True
                break
            n_read += 1
        if n_read < n_frames:
            truncated = True
        return n_read, n_frames, truncated


def read_dcd_trajectory(
    path: PathLike,
    *,
    max_frames: int | None = None,
    require_complete: bool = True,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Read frames from a CHARMM/NAMD DCD file.

    Parameters
    ----------
    require_complete
        If True (default), raise when the file ends before the header frame count.
    """
    path = Path(path)
    readable, header_frames, truncated = scan_dcd_frame_count(path)
    if readable == 0:
        raise ValueError(f"No frames read from DCD: {path}")
    if require_complete and truncated:
        raise ValueError(
            f"Truncated DCD at frame {readable}/{header_frames}: {path} "
            f"(file size {path.stat().st_size} bytes; dynamics may have aborted mid-write)"
        )
    limit = readable if require_complete else readable
    if max_frames is not None:
        limit = min(limit, int(max_frames))

    with path.open("rb") as f:
        n_frames, n_atoms, nsavc, dt, has_unitcell = _read_dcd_header(f)
        frames: list[np.ndarray] = []
        n_read = 0
        while n_read < limit:
            if has_unitcell:
                rec = struct.unpack("<i", f.read(4))[0]
                if rec == 0:
                    break
                f.read(rec)
                struct.unpack("<i", f.read(4))[0]
            struct.unpack("<i", f.read(4))[0]
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
        "header_n_frames": header_frames,
        "truncated": truncated,
    }
