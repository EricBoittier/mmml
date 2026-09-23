"""``concat_dcd_files`` robustness and DCD header unit-cell detection."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    count_dcd_frames,
    count_readable_dcd_frames,
)
from mmml.utils.dcd_writer import (
    _dcd_header_byte_size,
    concat_dcd_files,
    save_trajectory_dcd,
)

N_ATOMS = 3


def _write(path, n_frames, *, box=True, nsavc=125, value=0.0):
    pos = np.full((n_frames, N_ATOMS, 3), value, dtype=float)
    boxes = [np.array([30.0, 30.0, 30.0])] * max(1, n_frames) if box else None
    save_trajectory_dcd(path, pos, [None] * N_ATOMS, boxes=boxes, steps_per_frame=nsavc)
    return path


def _charmm_like_header(*, delta_bits: int, unitcell: int) -> bytes:
    """84-byte CORD record with ICNTRL(10)=DELTA and ICNTRL(11)=unit-cell flag."""
    icntrl = [0] * 20
    icntrl[0] = 0
    icntrl[9] = delta_bits
    icntrl[10] = unitcell
    icntrl[19] = 47
    cord = b"CORD" + struct.pack("<20i", *icntrl)
    title = struct.pack("<i", 1) + b"T".ljust(80)
    out = struct.pack("<i", len(cord)) + cord + struct.pack("<i", len(cord))
    out += struct.pack("<i", len(title)) + title + struct.pack("<i", len(title))
    out += struct.pack("<iii", 4, N_ATOMS, 4)
    return out


def test_unitcell_flag_read_from_icntrl11():
    # DELTA (ICNTRL(10)) is nonzero in every CHARMM file; it must not imply a cell.
    delta = struct.unpack("<i", struct.pack("<f", 0.0204545))[0]
    _, _, natoms, has_uc = _dcd_header_byte_size(_charmm_like_header(delta_bits=delta, unitcell=0))
    assert natoms == N_ATOMS
    assert has_uc is False
    _, _, _, has_uc = _dcd_header_byte_size(_charmm_like_header(delta_bits=delta, unitcell=1))
    assert has_uc is True


def test_unitcell_flag_for_mmml_writer(tmp_path):
    with_box = _write(tmp_path / "a.dcd", 1, box=True)
    no_box = _write(tmp_path / "b.dcd", 1, box=False)
    assert _dcd_header_byte_size(with_box.read_bytes())[3] is True
    assert _dcd_header_byte_size(no_box.read_bytes())[3] is False


def test_concat_skips_zero_byte_and_header_only(tmp_path):
    empty = tmp_path / "empty.dcd"
    empty.write_bytes(b"")
    header_only = _write(tmp_path / "hdr.dcd", 0)
    a = _write(tmp_path / "a.dcd", 2, value=1.0)
    b = _write(tmp_path / "b.dcd", 3, value=2.0)
    out = tmp_path / "out.dcd"

    n = concat_dcd_files([empty, header_only, a, tmp_path / "missing.dcd", b], out)

    assert n == 5
    assert count_dcd_frames(out) == 5
    assert count_readable_dcd_frames(out) == 5


def test_concat_uses_first_valid_file_as_header(tmp_path):
    empty = tmp_path / "empty.dcd"
    empty.write_bytes(b"")
    a = _write(tmp_path / "a.dcd", 2)
    out = tmp_path / "out.dcd"
    assert concat_dcd_files([empty, a], out) == 2
    assert count_readable_dcd_frames(out) == 2


def test_concat_all_invalid_removes_output(tmp_path):
    empty = tmp_path / "empty.dcd"
    empty.write_bytes(b"")
    out = tmp_path / "out.dcd"
    out.write_bytes(b"stale")
    assert concat_dcd_files([empty], out) == 0
    assert not out.exists()


def test_concat_unitcell_mismatch_raises(tmp_path):
    a = _write(tmp_path / "a.dcd", 2, box=True)
    b = _write(tmp_path / "b.dcd", 2, box=False)
    with pytest.raises(ValueError, match="unit-cell"):
        concat_dcd_files([a, b], tmp_path / "out.dcd")


def test_concat_header_nset_is_sum_of_frames(tmp_path):
    parts = [_write(tmp_path / f"p{i}.dcd", n, value=float(i)) for i, n in enumerate((2, 1, 4))]
    out = tmp_path / "out.dcd"
    assert concat_dcd_files(parts, out) == 7
    data = out.read_bytes()
    assert struct.unpack("<i", data[8:12])[0] == 7
    assert count_readable_dcd_frames(out) == 7
