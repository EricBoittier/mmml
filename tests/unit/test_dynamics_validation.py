"""Tests for post-dynamics DCD/restart validation."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    assert_stage_dynamics_completed,
    count_dcd_frames,
    expected_dcd_frame_count,
    read_restart_last_step,
)
from mmml.utils.dcd_writer import save_trajectory_dcd


def test_expected_dcd_frame_count():
    assert expected_dcd_frame_count(nstep=40000, nsavc=500) == 81
    assert expected_dcd_frame_count(nstep=721, nsavc=500) == 2


def test_count_dcd_frames(tmp_path):
    path = tmp_path / "t.dcd"
    import numpy as np

    class _Atoms:
        def __len__(self):
            return 2

    save_trajectory_dcd(
        path,
        np.zeros((3, 2, 3), dtype=np.float32),
        _Atoms(),
        steps_per_frame=10,
    )
    assert count_dcd_frames(path) == 3


def test_read_restart_last_step(tmp_path):
    res = tmp_path / "heat.res"
    res.write_text("REST     0    721\n")
    assert read_restart_last_step(res) == 721


def test_assert_stage_dynamics_completed_fails_truncated(tmp_path):
    dcd = tmp_path / "heat.dcd"
    res = tmp_path / "heat.res"
    res.write_text("REST     0    721\n")
    # Minimal valid DCD header claiming 1 frame
    with dcd.open("wb") as f:
        f.write(struct.pack("<i", 84))
        f.write(b"CORD")
        f.write(struct.pack("<i", 1))
        f.write(b"\x00" * 72)
        f.write(struct.pack("<i", 164))
        f.write(struct.pack("<i", 2))
        f.write(b"x" * 160)
        f.write(struct.pack("<i", 164))
        f.write(struct.pack("<i", 4))
        f.write(struct.pack("<i", 45))
        f.write(struct.pack("<i", 4))

    with pytest.raises(RuntimeError, match="HEAT dynamics incomplete"):
        assert_stage_dynamics_completed(
            stage="heat",
            expected_nstep=40000,
            nsavc=500,
            dcd_path=dcd,
            restart_path=res,
        )
