"""Tests for post-dynamics DCD/restart validation."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    assert_stage_dynamics_completed,
    count_dcd_frames,
    expected_dcd_frame_count,
    read_restart_coordinates,
    read_restart_last_step,
    read_restart_natom,
    restart_has_nonfinite_coordinates,
    _restart_coordinate_values,
)
from mmml.utils.dcd_writer import concat_dcd_files, save_trajectory_dcd


def test_expected_dcd_frame_count():
    assert expected_dcd_frame_count(nstep=40000, nsavc=500) == 81
    assert expected_dcd_frame_count(nstep=721, nsavc=500) == 2
    assert expected_dcd_frame_count(nstep=1, nsavc=1) == 1


def test_harmonize_nsavc_frequency():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        harmonize_nsavc_frequency,
    )

    assert harmonize_nsavc_frequency(100, 41) == 1
    assert harmonize_nsavc_frequency(40, 40) == 20
    assert harmonize_nsavc_frequency(10, 40) == 10
    assert harmonize_nsavc_frequency(200, 500) == 125


def test_expected_overlap_chunk_dcd_frame_count_benz30_equi_case():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        expected_overlap_chunk_dcd_frame_count,
    )

    # 5000 steps, 10 × 500-step overlap chunks, nsavc=200 → 2 frames/chunk (restart).
    assert (
        expected_overlap_chunk_dcd_frame_count(
            total_nstep=5000, nsavc=200, n_chunks=10
        )
        == 20
    )


def test_assert_stage_dynamics_completed_accepts_overlap_chunk_dcds(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )

    class _Atoms:
        def __len__(self):
            return 5

    atoms = _Atoms()
    dcd = tmp_path / "equi_dcm_200.dcd"
    for i in range(10):
        save_trajectory_dcd(
            tmp_path / f"equi_dcm_200.chunk.{i:04d}.dcd",
            np.zeros((2, 5, 3), dtype=np.float32),
            atoms,
            steps_per_frame=200,
        )
    res = tmp_path / "equi.res"
    res.write_text(
        "REST     0         5000\n!NATOM\n10\n!X, Y, Z\n"
        + " ".join(["0.0"] * 30)
        + "\n",
        encoding="utf-8",
    )
    assert_stage_dynamics_completed(
        stage="equi",
        expected_nstep=5000,
        nsavc=200,
        dcd_path=dcd,
        restart_path=res,
    )


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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._release_charmm_dynamics_api_buffers"
    ) as release_bufs, patch.dict(sys.modules, {"pycharmm": fake_pycharmm}):
        run_dynamics({"iasvel": 0, "start": False, "nstep": 10})
    clear_comp.assert_called_once()
    dyn.run.assert_called_once()
    assert release_bufs.call_count == 2


def test_release_charmm_dynamics_api_buffers_calls_del_routines():
    import sys
    from unittest.mock import MagicMock, patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _release_charmm_dynamics_api_buffers,
    )

    fake_lib = MagicMock()
    with patch.dict(sys.modules, {"pycharmm.lib": fake_lib}, clear=False):
        _release_charmm_dynamics_api_buffers()
    fake_lib.charmm.lambdata_del.assert_called_once()
    fake_lib.charmm.ktable_del.assert_called_once()
    fake_lib.charmm.velos_del.assert_called_once()
    fake_lib.charmm.msldata_del.assert_called_once()


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


def test_integrated_step_from_restart_stale_nstep_field(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _integrated_step_from_restart,
    )

    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     0\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          25         500         500         500          10           0\n",
        encoding="utf-8",
    )
    io = CharmmTrajectoryFiles(restart_write=res)
    assert (
        _integrated_step_from_restart(
            chunk_io=io,
            final_restart=res,
            fallback_steps=2500,
        )
        == 2500
    )


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
        f"{'25':>10}{'8000':>10}{'500':>10}{'500':>10}{'10':>10}{'0':>10}\n",
        encoding="utf-8",
    )
    assert patch_restart_global_step(res, 1500)
    assert read_restart_last_step(res) == 1500


def test_patch_restart_global_step_preserves_fortran_restart_format(tmp_path):
    """Post-rescue READYN needs fixed-width I10 lines, not space-joined tokens."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        patch_restart_global_step,
    )

    stub = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/output/dynamics/nve_stub.res"
    )
    assert stub.is_file()
    res = tmp_path / "overlap_a.res"
    res.write_text(stub.read_text(encoding="utf-8"), encoding="utf-8")
    natom_line_before = res.read_text(encoding="utf-8").splitlines()[7]
    tail_before = natom_line_before[60:]

    assert patch_restart_global_step(res, 500)

    lines = res.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("REST")
    assert lines[0][10:20] == "       500"
    natom_line = lines[7]
    assert natom_line[50:60] == "       500"
    assert natom_line[60:] == tail_before


def test_read_restart_last_step_real_fixture():
    stub = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/output/dynamics/nve_stub.res"
    )
    assert stub.is_file()
    assert read_restart_last_step(stub) == 2000


def test_read_restart_coordinates_from_fixture():
    stub = (
        Path(__file__).resolve().parents[1]
        / "functionality/mlpot/output/dynamics/nve_stub.res"
    )
    assert stub.is_file()
    assert read_restart_natom(stub) == 20
    pos = read_restart_coordinates(stub)
    assert pos is not None
    assert pos.shape == (20, 3)
    assert np.all(np.isfinite(pos))
    assert pos[0, 0] == pytest.approx(1.38193103193303e-3)
    assert pos[0, 1] == pytest.approx(1.13107354438552e-3)
    assert pos[0, 2] == pytest.approx(2.29160874829407e-3)
    # Concatenated Fortran floats on line 537 of the fixture (atom 2).
    assert pos[1, 0] == pytest.approx(-9.35100401619100e-4)
    assert pos[1, 1] == pytest.approx(2.56810137565518e-4)
    assert pos[1, 2] == pytest.approx(-7.53960710226643e-4)


def _minimal_restart_text(*, natom: int, coord_lines: list[str]) -> str:
    return (
        "REST     0     1\n"
        "       2 !NTITLE followed by title\n"
        "* minimal restart for unit tests\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        f"         {natom}           0           0           0           0           0           0\n"
        "\n"
        " !X, Y, Z\n"
        + "\n".join(coord_lines)
        + "\n"
    )


def test_restart_coordinate_values_splits_concatenated_fortran_floats(tmp_path):
    """Whitespace split misses glued tokens like ``D-03-0.753...``."""
    path = tmp_path / "concat.res"
    path.write_text(
        _minimal_restart_text(
            natom=2,
            coord_lines=[
                " 0.256810137565518D-03-0.753960710226643D-03 0.100000000000000D+00",
                " 0.200000000000000D+00 0.300000000000000D+00 0.400000000000000D+00",
            ],
        )
    )
    flat = _restart_coordinate_values(path)
    assert len(flat) == 6
    assert flat[0] == pytest.approx(2.56810137565518e-4)
    assert flat[1] == pytest.approx(-7.53960710226643e-4)
    assert flat[2] == pytest.approx(0.1)

    pos = read_restart_coordinates(path)
    assert pos is not None
    assert pos.shape == (2, 3)
    assert np.allclose(pos[0], flat[:3])
    assert np.allclose(pos[1], flat[3:])


def test_read_restart_coordinates_returns_none_when_too_few_values(tmp_path):
    path = tmp_path / "short.res"
    path.write_text(
        _minimal_restart_text(
            natom=3,
            coord_lines=[" 0.100000000000000D+00 0.200000000000000D+00"],
        )
    )
    assert read_restart_coordinates(path) is None


def test_restart_has_nonfinite_coordinates_detects_nan_in_coord_section(tmp_path):
    path = tmp_path / "nan.res"
    path.write_text(
        _minimal_restart_text(
            natom=1,
            coord_lines=[" 0.100000000000000D+00          NaN 0.300000000000000D+00"],
        )
    )
    assert read_restart_coordinates(path) is None
    assert restart_has_nonfinite_coordinates(path) is True


def test_restart_has_nonfinite_coordinates_false_for_finite_restart(tmp_path):
    path = tmp_path / "ok.res"
    path.write_text(
        _minimal_restart_text(
            natom=1,
            coord_lines=[" 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00"],
        )
    )
    assert restart_has_nonfinite_coordinates(path) is False


def test_assert_stage_dynamics_completed_accepts_single_step_heat(tmp_path):
    """nstep=1 heat (instant velocity scaling) writes one DCD frame, not two."""
    dcd = tmp_path / "heat.dcd"
    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          10     0       1       1      10     500     297       0       0\n",
        encoding="utf-8",
    )

    class _Atoms:
        def __len__(self):
            return 5

    save_trajectory_dcd(
        dcd,
        np.zeros((1, 5, 3), dtype=np.float32),
        _Atoms(),
        steps_per_frame=1,
    )
    assert_stage_dynamics_completed(
        stage="heat",
        expected_nstep=1,
        nsavc=1,
        dcd_path=dcd,
        restart_path=res,
    )


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


def test_restart_has_nonfinite_coordinates_detects_nan_section(tmp_path):
    res = tmp_path / "bad.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          20        2000        1000           1          10        1000\n"
        "\n"
        " !X, Y, Z\n"
        " NaN 0.0 0.0\n"
        " 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    assert restart_has_nonfinite_coordinates(res) is True


def test_assert_stage_dynamics_completed_fails_nonfinite_restart(tmp_path):
    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     1\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          20        2000        1000           1          10        1000\n"
        "\n"
        " !X, Y, Z\n"
        " NaN 0.0 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-finite coordinates"):
        assert_stage_dynamics_completed(
            stage="heat",
            expected_nstep=1000,
            nsavc=100,
            dcd_path=None,
            restart_path=res,
        )


def test_resolve_integrated_restart_step_stale_subchunk_header(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        resolve_integrated_restart_step,
    )

    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     0\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          25         500         500         500          10           0\n",
        encoding="utf-8",
    )
    assert resolve_integrated_restart_step(res, expected_nstep=2500) == 2500


def test_assert_stage_dynamics_completed_accepts_stale_restart_after_rescue(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        assert_stage_dynamics_completed,
    )

    res = tmp_path / "heat.res"
    res.write_text(
        "REST    48     0\n"
        "\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "          25         500         500         500          10           0\n",
        encoding="utf-8",
    )

    assert_stage_dynamics_completed(
        stage="heat",
        expected_nstep=2500,
        nsavc=100,
        dcd_path=None,
        restart_path=res,
    )


def test_rewrite_dynamics_restart_validated_detects_nan(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        rewrite_dynamics_restart_validated,
    )

    bad = tmp_path / "heat.res"
    good = tmp_path / "good.res"
    good.write_text(
        _minimal_restart_text(
            natom=1,
            coord_lines=[" 0.100000000000000D+00 0.200000000000000D+00 0.300000000000000D+00"],
        ),
        encoding="utf-8",
    )
    bad.write_text(
        _minimal_restart_text(
            natom=1,
            coord_lines=[" NaN 0.200000000000000D+00 0.300000000000000D+00"],
        ),
        encoding="utf-8",
    )

    def fake_rewrite(path, *, write_unit=92):
        Path(path).write_text(bad.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_from_current_state",
        fake_rewrite,
    )
    assert rewrite_dynamics_restart_validated(bad) is False


def test_restore_post_rescue_coordinates_prefers_memory(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        restore_post_rescue_coordinates,
    )

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    synced: list[np.ndarray] = []

    def fake_sync(arr):
        synced.append(np.asarray(arr, dtype=float).copy())

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        fake_sync,
    )
    source = restore_post_rescue_coordinates(rescued_positions=pos)
    assert source == "in-memory rescue positions"
    assert np.allclose(synced[0], pos)


def test_refresh_segment_restart_after_overlap_rescue_recovers_from_memory(
    tmp_path, monkeypatch, capsys
):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _refresh_segment_restart_after_overlap_rescue,
    )

    restart = tmp_path / "heat.res"
    rescued = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    validated_calls = {"n": 0}

    def fake_validated(path, *, write_unit=92):
        validated_calls["n"] += 1
        return validated_calls["n"] > 1

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
        fake_validated,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        lambda: rescued.copy(),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._assign_post_rescue_velocities_and_crystal",
        lambda *_a, **_kw: None,
    )
    restored: list[np.ndarray] = []

    def fake_restore(**kwargs):
        restored.append(kwargs["rescued_positions"])
        return "in-memory rescue positions"

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_post_rescue_coordinates",
        fake_restore,
    )

    _refresh_segment_restart_after_overlap_rescue(
        restart,
        {"firstt": 57.0, "tbath": 57.0},
        mlpot_ctx=None,
    )
    out = capsys.readouterr().out
    assert validated_calls["n"] == 2
    assert restored and np.allclose(restored[0], rescued)
    assert "restored from in-memory rescue positions" in out
    assert "post-rescue restart recovered" in out
