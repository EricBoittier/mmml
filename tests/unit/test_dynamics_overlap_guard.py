"""Tests for PyCHARMM MLpot dynamics overlap guard."""

from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    DynamicsOverlapConfig,
    OverlapRescueConfig,
    check_dynamics_overlap,
    monomer_offsets,
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics_with_io


def test_monomer_offsets_uniform():
    off = monomer_offsets(10, 2)
    np.testing.assert_array_equal(off, [0, 5, 10])


def test_resolve_defaults_to_rescue_and_1p5A():
    args = argparse.Namespace()
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=True)
    assert cfg.action == "rescue"
    assert cfg.min_distance_A == 1.5
    assert cfg.check_interval == 50
    assert cfg.enabled is True
    assert cfg.rescue.nstep_sd == 200
    assert cfg.rescue.nstep_abnr == 400


def test_resolve_off_disables():
    args = argparse.Namespace(dynamics_overlap_action="off")
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=False)
    assert cfg.enabled is False


def test_resolve_pbc_stores_box_size_fallback():
    args = argparse.Namespace(box_size=30.0)
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=True)
    assert cfg.fallback_box_side_A == pytest.approx(30.0)


def test_overlap_cell_uses_fallback_when_pbound_zero():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env._read_charmm_box_sides_A",
        return_value=(0.0, 0.0, 0.0),
    ):
        cfg = DynamicsOverlapConfig(
            action="error",
            min_distance_A=1.5,
            n_monomers=2,
            use_pbc=True,
            fallback_box_side_A=30.0,
        )
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                [16.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        with mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            return_value=pos,
        ):
            dmin = check_dynamics_overlap(cfg, context="test", step=0)
    assert dmin > 1.5


def test_check_overlap_raises_on_close_contact():
    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=1.5,
        n_monomers=2,
        use_pbc=False,
    )
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos,
    ):
        with pytest.raises(RuntimeError, match="inter-monomer atom overlap"):
            check_dynamics_overlap(cfg, context="test", step=50)


def test_check_overlap_rescue_runs_minimize_and_rechecks():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        n_monomers=2,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=10, nstep_abnr=0, verbose=False),
    )
    pos_bad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    ctx = mock.Mock()
    call_n = [0]

    def get_pos():
        call_n[0] += 1
        return pos_bad if call_n[0] == 1 else pos_ok

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        side_effect=get_pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_overlap_rescue",
    ) as rescue:
        dmin = check_dynamics_overlap(
            cfg, context="test", step=50, mlpot_ctx=ctx
        )
        rescue.assert_called_once_with(ctx, cfg.rescue)
    assert dmin > 1.5


def test_run_dynamics_with_io_chunks_and_checks(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=2,
        n_monomers=2,
        use_pbc=False,
    )
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    res_path = tmp_path / "equi.res"
    res_path.write_text("REST    48     1  CUBI\n", encoding="utf-8")
    io = CharmmTrajectoryFiles(
        restart_read=tmp_path / "heat.res",
        restart_write=res_path,
    )
    (tmp_path / "heat.res").write_text("REST    48     1  CUBI\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_chunk(kw, _io):
        calls.append(dict(kw))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {
                "nstep": 5,
                "new": True,
                "start": True,
                "restart": True,
                "iunrea": 3,
                "firstt": 240.0,
            },
            io,
            overlap=cfg,
            overlap_context="NVE",
        )

    assert [c["nstep"] for c in calls] == [2, 2, 1]
    assert calls[0]["restart"] is True
    assert calls[0]["iunrea"] == 3
    assert "firstt" in calls[0]
    assert calls[1]["restart"] is False
    assert calls[1]["iunrea"] == -1
    assert "firstt" not in calls[1]
    assert calls[2]["restart"] is False
    assert calls[2]["iunrea"] == -1
    assert "firstt" not in calls[2]


def test_overlap_memory_handoff_chunks_no_restart_read(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=2,
        n_monomers=2,
        use_pbc=False,
    )
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    res_path = tmp_path / "equi.res"
    io = CharmmTrajectoryFiles(restart_write=res_path)
    calls: list[dict] = []

    def fake_chunk(kw, _io):
        calls.append(dict(kw))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {
                "nstep": 5,
                "new": False,
                "start": False,
                "restart": False,
            },
            io,
            overlap=cfg,
            overlap_context="NVE",
        )

    assert [c["nstep"] for c in calls] == [2, 2, 1]
    for c in calls:
        assert c["restart"] is False
        assert c.get("iunrea") == -1
        assert "firstt" not in c


def test_overlap_cleans_stale_slots_at_start(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_restart_slot_paths,
        _cleanup_overlap_restart_slots,
    )

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=2,
        n_monomers=2,
        use_pbc=False,
    )
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    equi = tmp_path / "equi.res"
    io = CharmmTrajectoryFiles(restart_write=equi)
    slot_a, slot_b = _overlap_restart_slot_paths(equi)
    slot_a.write_text("", encoding="utf-8")

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        return_value=mock.Mock(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 4, "restart": False},
            io,
            overlap=cfg,
        )

    assert not slot_a.exists()
    _cleanup_overlap_restart_slots(io)
    assert not slot_b.exists()


def test_overlap_chunk_io_external_read_last_write_only(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_chunk_io,
    )

    heat = tmp_path / "heat.res"
    equi = tmp_path / "equi_dcm_60.res"
    heat.write_text("REST\n", encoding="utf-8")
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)

    c0 = _overlap_chunk_io(io, chunk_index=0, n_chunks=3)
    assert c0.restart_read == heat
    assert c0.restart_write is None

    c1 = _overlap_chunk_io(io, chunk_index=1, n_chunks=3)
    assert c1.restart_read is None
    assert c1.restart_write is None
    assert c1.trajectory == io.trajectory

    c2 = _overlap_chunk_io(io, chunk_index=2, n_chunks=3)
    assert c2.restart_read is None
    assert c2.restart_write == equi
    assert c2.trajectory == io.trajectory


def test_run_dynamics_chunk_strips_stale_iunwri(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_dynamics_chunk,
    )

    io = CharmmTrajectoryFiles(
        restart_write=tmp_path / "out.res",
        trajectory=tmp_path / "out.dcd",
    )
    captured: list[dict] = []

    def fake_run(kw):
        captured.append(dict(kw))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics",
        side_effect=fake_run,
    ), mock.patch.object(
        CharmmTrajectoryFiles,
        "open_for_run",
        return_value=([], {}),
    ):
        _run_dynamics_chunk(
            {"nstep": 10, "restart": True, "iunwri": 2, "iunrea": 3},
            io,
        )
    assert "iunwri" not in captured[0]
    assert "iunrea" not in captured[0]
