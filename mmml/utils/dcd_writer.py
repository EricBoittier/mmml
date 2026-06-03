"""
DCD trajectory writer for CHARMM compatibility.

Pure Python implementation of the CHARMM/NAMD DCD binary format.
No external dependencies beyond NumPy and the standard library.
"""

from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

# DCD uses 32-bit floats for coordinates, little-endian
_ENDIAN = "<"
_FMT_I = _ENDIAN + "i"
_FMT_F = _ENDIAN + "f"
_FMT_D = _ENDIAN + "d"


def _box_to_dcd_unitcell(box: np.ndarray) -> np.ndarray:
    """Convert box (3,3) or (3,) to DCD unit cell format [A, cos(gamma), B, cos(beta), cos(alpha), C].

    For orthorhombic boxes, angles are 90° so cosines are 0.
    """
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 1 and box.size >= 3:
        a, b, c = float(box[0]), float(box[1]), float(box[2])
        return np.array([a, 0.0, b, 0.0, 0.0, c], dtype=np.float64)
    if box.ndim == 2 and box.shape[0] >= 3 and box.shape[1] >= 3:
        # Full 3x3 cell: rows are a, b, c vectors
        h = box[:3, :3]
        a_vec, b_vec, c_vec = h[0], h[1], h[2]
        A = float(np.linalg.norm(a_vec))
        B = float(np.linalg.norm(b_vec))
        C = float(np.linalg.norm(c_vec))
        if A < 1e-10 or B < 1e-10 or C < 1e-10:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Angles: alpha = angle(b,c), beta = angle(a,c), gamma = angle(a,b)
        cos_alpha = float(np.dot(b_vec, c_vec) / (B * C))
        cos_beta = float(np.dot(a_vec, c_vec) / (A * C))
        cos_gamma = float(np.dot(a_vec, b_vec) / (A * B))
        # Clamp to [-1, 1] for numerical safety
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        cos_beta = max(-1.0, min(1.0, cos_beta))
        cos_gamma = max(-1.0, min(1.0, cos_gamma))
        # DCD NAMD/VMD format: [A, cos(gamma), B, cos(beta), cos(alpha), C]
        return np.array([A, cos_gamma, B, cos_beta, cos_alpha, C], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def save_trajectory_dcd(
    path: Union[str, Path],
    positions: np.ndarray,
    atoms: Any,
    boxes: Optional[List[Any]] = None,
    dt_ps: Optional[float] = None,
    steps_per_frame: int = 1,
) -> None:
    """Save trajectory to a CHARMM-readable DCD file.

    Pure Python implementation; no MDAnalysis or other optional dependencies.

    Parameters
    ----------
    path : str | Path
        Output DCD file path.
    positions : np.ndarray
        Atomic positions, shape (n_frames, n_atoms, 3) in Å.
    atoms : ase.Atoms
        ASE Atoms object (used only for n_atoms; topology not stored in DCD).
    boxes : list of np.ndarray | None
        Optional box per frame for NPT. Each element is (3,3) or (3,) in Å.
    dt_ps : float | None
        Timestep in ps between frames. If None, uses 1.0 (DCD default).
    steps_per_frame : int
        MD steps between recorded frames (nsavc in DCD header). Default 1.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1] != len(atoms) or positions.shape[2] != 3:
        positions = positions.reshape(-1, len(atoms), 3)

    n_frames = positions.shape[0]
    n_atoms = positions.shape[1]
    dt = float(dt_ps) if dt_ps is not None else 1.0
    has_unitcell = boxes is not None and len(boxes) > 0

    with open(path, "wb") as f:
        # ----- Header block (84 bytes) -----
        f.write(struct.pack(_FMT_I, 84))
        f.write(b"CORD")
        f.write(struct.pack(_FMT_I, n_frames))
        f.write(struct.pack(_FMT_I, 0))  # istart
        f.write(struct.pack(_FMT_I, steps_per_frame))
        f.write(struct.pack(_FMT_I, 0))  # numsteps
        f.write(struct.pack(_FMT_I, 0))
        f.write(struct.pack(_FMT_I, 0) * 4)  # 5 zeros
        f.write(struct.pack(_FMT_F, dt))
        f.write(struct.pack(_FMT_I, 1 if has_unitcell else 0))
        f.write(struct.pack(_FMT_I, 0) * 8)
        f.write(struct.pack(_FMT_I, 24))  # CHARMM version 24
        f.write(struct.pack(_FMT_I, 84))

        # ----- Title block (164 bytes) -----
        f.write(struct.pack(_FMT_I, 164))
        f.write(struct.pack(_FMT_I, 2))
        title1 = b"Created by mmml DCD writer".ljust(80)
        title2 = datetime.now().strftime("%d %B, %Y at %H:%M").encode().ljust(80)
        f.write(title1)
        f.write(title2)
        f.write(struct.pack(_FMT_I, 164))

        # ----- Atom count block -----
        f.write(struct.pack(_FMT_I, 4))
        f.write(struct.pack(_FMT_I, n_atoms))
        f.write(struct.pack(_FMT_I, 4))

        # ----- Frames -----
        block_size_xyz = n_atoms * 4
        for i in range(n_frames):
            if has_unitcell:
                box = np.asarray(boxes[i]) if i < len(boxes) else np.asarray(boxes[-1])
                uc = _box_to_dcd_unitcell(box)
                f.write(struct.pack(_FMT_I, 48))
                uc.astype(np.float64).tofile(f)
                f.write(struct.pack(_FMT_I, 48))

            xyz = positions[i]
            for j in range(3):
                f.write(struct.pack(_FMT_I, block_size_xyz))
                xyz[:, j].astype(np.float32).tofile(f)
                f.write(struct.pack(_FMT_I, block_size_xyz))


def _dcd_header_byte_size(data: bytes) -> tuple[int, int, int, bool]:
    """Parse a DCD header; return ``(header_bytes, n_frames, n_atoms, has_unitcell)``."""
    offset = 0
    n_frames = 0
    has_unitcell = False
    n_atoms = 0

    def read_record() -> bytes:
        nonlocal offset
        if offset + 4 > len(data):
            raise ValueError("truncated DCD record")
        rec_size = struct.unpack(_FMT_I, data[offset : offset + 4])[0]
        offset += 4
        end = offset + rec_size
        if end + 4 > len(data):
            raise ValueError("truncated DCD record payload")
        payload = data[offset:end]
        offset = end + 4
        return payload

    cord_block = read_record()
    if cord_block[:4] != b"CORD":
        raise ValueError("invalid DCD header (missing CORD magic)")
    n_frames = struct.unpack(_FMT_I, cord_block[4:8])[0]
    if len(cord_block) >= 44:
        has_unitcell = struct.unpack(_FMT_I, cord_block[40:44])[0] != 0
    read_record()  # title
    natoms_block = read_record()
    if len(natoms_block) != 4:
        raise ValueError(f"unexpected DCD natoms record size {len(natoms_block)}")
    n_atoms = struct.unpack(_FMT_I, natoms_block)[0]
    return offset, int(n_frames), int(n_atoms), has_unitcell


def _dcd_frame_byte_size(n_atoms: int, *, has_unitcell: bool) -> int:
    xyz = 3 * (4 + 4 * n_atoms + 4)
    if has_unitcell:
        return 4 + 48 + 4 + xyz
    return xyz


def concat_dcd_files(chunks: list[Union[str, Path]], out: Union[str, Path]) -> int:
    """Concatenate same-system DCD chunks; return total frame count.

    Chunk files must share atom count and unit-cell flag. The first chunk supplies
    the header; frame blocks from every chunk are appended in order.
    """
    paths = [Path(p) for p in chunks if Path(p).is_file()]
    out_path = Path(out)
    if not paths:
        out_path.unlink(missing_ok=True)
        return 0
    if len(paths) == 1:
        out_path.write_bytes(paths[0].read_bytes())
        with out_path.open("rb") as f:
            _, n_frames, _, _ = _dcd_header_byte_size(f.read(4096))
        return n_frames

    first = paths[0].read_bytes()
    header_end, _, n_atoms, has_unitcell = _dcd_header_byte_size(first)
    frame_bytes = _dcd_frame_byte_size(n_atoms, has_unitcell=has_unitcell)
    total_frames = 0
    frame_blob = bytearray()
    for path in paths:
        data = path.read_bytes()
        hdr_end, n_frames, natoms, uc = _dcd_header_byte_size(data)
        if natoms != n_atoms or uc != has_unitcell:
            raise ValueError(
                f"DCD chunk {path.name} incompatible with {paths[0].name} "
                f"(natoms {natoms} vs {n_atoms}, unitcell {uc} vs {has_unitcell})"
            )
        payload = data[hdr_end:]
        expected = n_frames * frame_bytes
        if len(payload) < expected:
            raise ValueError(
                f"DCD chunk {path.name} truncated: expected {expected} frame bytes, "
                f"got {len(payload)}"
            )
        frame_blob.extend(payload[:expected])
        total_frames += n_frames

    out_bytes = bytearray(first[:header_end])
    out_bytes[8:12] = struct.pack(_FMT_I, total_frames)
    out_bytes.extend(frame_blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out_bytes)
    return total_frames
