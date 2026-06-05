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
from mmml.utils.dcd_writer import concat_dcd_files, save_trajectory_dcd


def test_expected_dcd_frame_count():
    assert expected_dcd_frame_count(nstep=40000, nsavc=500) == 81
    assert expected_dcd_frame_count(nstep=721, nsavc=500) == 2


def test_scan_dcd_frame_count_truncated_header(tmp_path):
    from mmml.utils.dcd_reader import scan_dcd_frame_count
    from mmml.utils.dcd_writer import save_trajectory_dcd

    import numpy as np

    class _Atoms:
        def __len__(self):
            return 5

    path = tmp_path / "nve.dcd"
    save_trajectory_dcd(
        path,
        np.zeros((2, 5, 3), dtype=np.float32),
        _Atoms(),
        steps_per_frame=1,
    )
    data = bytearray(path.read_bytes())
    data[8:12] = struct.pack("<i", 80000)
    path.write_bytes(data)

    readable, header, truncated = scan_dcd_frame_count(path)
    assert header == 80000
    assert readable == 2
    assert truncated is True


def test_count_readable_dcd_frames_truncated(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        count_readable_dcd_frames,
    )
    from mmml.utils.dcd_writer import save_trajectory_dcd

    import numpy as np

    class _Atoms:
        def __len__(self):
            return 5

    path = tmp_path / "nve.dcd"
    save_trajectory_dcd(
        path,
        np.zeros((3, 5, 3), dtype=np.float32),
        _Atoms(),
        steps_per_frame=1,
    )
    data = bytearray(path.read_bytes())
    data[8:12] = struct.pack("<i", 1000)
    path.write_bytes(data)

    assert count_readable_dcd_frames(path) == 3


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


def test_run_dynamics_clears_comparison_coords_when_iasvel_zero_no_start():
    import sys
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics

    fake_pycharmm = MagicMock()
    dyn = MagicMock()
    fake_pycharmm.DynamicsScript.return_value = dyn
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.comp_velocities.clear_comparison_coordinates"
    ) as clear_comp, patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.disable_charmm_domdec"
    ), patch.dict(sys.modules, {"pycharmm": fake_pycharmm}):
        run_dynamics({"iasvel": 0, "start": False, "nstep": 10})
    clear_comp.assert_called_once()
    dyn.run.assert_called_once()


def test_build_nve_dynamics_restart_uses_iasvel_zero():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import build_nve_dynamics

    kw = build_nve_dynamics(restart=True, temp=300.0)
    assert kw["iasvel"] == 0
    assert "firstt" not in kw
    cold = build_nve_dynamics(restart=False, temp=300.0)
    assert cold["iasvel"] == 1
    assert cold["firstt"] == 300.0


def test_concat_dcd_files_merges_chunks(tmp_path):
    import numpy as np

    class _Atoms:
        def __len__(self):
            return 2

    atoms = _Atoms()
    c1 = tmp_path / "heat.chunk.0000.dcd"
    c2 = tmp_path / "heat.chunk.0001.dcd"
    out = tmp_path / "heat.dcd"
    save_trajectory_dcd(
        c1,
        np.zeros((2, 2, 3), dtype=np.float32),
        atoms,
        steps_per_frame=1,
    )
    save_trajectory_dcd(
        c2,
        np.ones((3, 2, 3), dtype=np.float32),
        atoms,
        steps_per_frame=1,
    )
    n = concat_dcd_files([c1, c2], out)
    assert n == 5
    assert count_dcd_frames(out) == 5


def test_concat_dcd_files_accepts_header_with_incomplete_last_frame(tmp_path):
    import numpy as np

    class _Atoms:
        def __len__(self):
            return 2

    atoms = _Atoms()
    chunk = tmp_path / "nve.chunk.0000.dcd"
    out = tmp_path / "nve.dcd"
    save_trajectory_dcd(
        chunk,
        np.zeros((1, 2, 3), dtype=np.float32),
        atoms,
        steps_per_frame=1,
    )
    data = bytearray(chunk.read_bytes())
    data[8:12] = struct.pack("<i", 2)
    chunk.write_bytes(data)
    n = concat_dcd_files([chunk], out)
    assert n == 1
    assert count_dcd_frames(out) == 1


def test_read_restart_last_step(tmp_path):
    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         100       10000         721         100          10         721\n"
    )
    assert read_restart_last_step(res) == 721


def test_read_restart_last_step_uses_nstep_when_jhstrt_zero(tmp_path):
    """CHARMM coord-history restarts: JHSTRT=0 but NSTEP=8000 for one segment."""
    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     0\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "         25       10000        8000         500          10           0\n",
        encoding="utf-8",
    )
    assert read_restart_last_step(res) == 8000


def test_read_restart_last_step_prefers_jhstrt_over_segment_nstep(tmp_path):
    """Overlap chunks: NSTEP=500 (last segment) but JHSTRT=8000 (global step)."""
    res = tmp_path / "nve.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          25        8000         500         500          10        8000\n"
    )
    assert read_restart_last_step(res) == 8000


def test_patch_restart_global_step_updates_jhstrt(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
        read_restart_last_step,
    )

    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     0\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          25        8000         500         500          10           0\n",
        encoding="utf-8",
    )
    assert patch_restart_global_step(res, 1500)
    assert read_restart_last_step(res) == 1500


def test_read_restart_last_step_real_fixture():
    stub = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/output/dynamics/nve_stub.res"
    )
    assert stub.is_file()
    assert read_restart_last_step(stub) == 2000


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
