"""Tests for PyCHARMM MLpot dynamics overlap guard."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    assert cfg.check_interval == 500
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


def test_check_overlap_rescue_applies_separation_last_resort():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        n_monomers=2,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=10, nstep_abnr=0, verbose=False),
        separate_on_rescue_fail=True,
        separate_margin_A=0.0,
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
        if call_n[0] <= 2:
            return pos_bad.copy()
        return pos_ok.copy()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        side_effect=get_pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
    ) as sync_pos, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.minimize_overlap_rescue",
    ) as rescue:
        dmin = check_dynamics_overlap(
            cfg, context="test", step=50, mlpot_ctx=ctx
        )
        rescue.assert_called_once_with(ctx, cfg.rescue)
        sync_pos.assert_called_once()
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

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {
                "nstep": 6,
                "new": True,
                "start": True,
                "restart": True,
                "iunrea": 3,
                "firstt": 240.0,
                "ihbfrq": 50,
                "imgfrq": 50,
            },
            io,
            overlap=cfg,
            overlap_context="NVE",
        )

    assert [c["nstep"] for c in calls] == [2, 2, 2]
    assert calls[0]["restart"] is True
    assert calls[0]["iunrea"] == 3
    assert "firstt" in calls[0]
    assert calls[1]["restart"] is True
    assert "firstt" not in calls[1]
    assert calls[2]["restart"] is True
    assert "firstt" not in calls[2]


def test_overlap_memory_handoff_chunks_scratch_restart_handoff(tmp_path):
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

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST\n", encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {
                "nstep": 6,
                "new": False,
                "start": False,
                "restart": False,
            },
            io,
            overlap=cfg,
            overlap_context="NVE",
        )

    assert [c["nstep"] for c in calls] == [2, 2, 2]
    assert calls[0]["restart"] is False
    assert calls[0].get("iunrea") == -1
    assert calls[1]["restart"] is True
    assert calls[2]["restart"] is True
    assert "firstt" not in calls[1]


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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
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


def test_overlap_chunk_io_alternate_scratch_and_final_write(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_chunk_io,
        _overlap_chunk_restart_paths,
        _overlap_restart_slot_paths,
    )

    heat = tmp_path / "heat.res"
    equi = tmp_path / "equi_dcm_60.res"
    heat.write_text("REST\n", encoding="utf-8")
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    slot_a, slot_b = _overlap_restart_slot_paths(equi)

    r0, w0 = _overlap_chunk_restart_paths(io, chunk_index=0, n_chunks=3)
    assert r0 == heat
    assert w0 == slot_a

    r1, w1 = _overlap_chunk_restart_paths(io, chunk_index=1, n_chunks=3)
    assert r1 == slot_a
    assert w1 == slot_b

    r2, w2 = _overlap_chunk_restart_paths(io, chunk_index=2, n_chunks=3)
    assert r2 == slot_b
    assert w2 == equi

    c0 = _overlap_chunk_io(io, chunk_index=0, n_chunks=3)
    assert c0.restart_read == heat
    assert c0.restart_write == slot_a

    c1 = _overlap_chunk_io(io, chunk_index=1, n_chunks=3)
    assert c1.restart_read == slot_a
    assert c1.restart_write == slot_b

    c2 = _overlap_chunk_io(io, chunk_index=2, n_chunks=3)
    assert c2.restart_read == slot_b
    assert c2.restart_write == equi
    assert c2.trajectory is None


def test_overlap_multi_chunk_keeps_dcd_open_across_chunks(tmp_path):
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
    dcd = tmp_path / "heat.dcd"
    io = CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res", trajectory=dcd)
    dcd_calls: list[int | None] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        merged = dict(kw)
        if extra_iokw:
            merged.update(extra_iokw)
        dcd_calls.append(merged.get("iuncrd"))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST\n", encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch.object(
        CharmmTrajectoryFiles,
        "open_trajectory_for_run",
        return_value=([mock.Mock()], {"iuncrd": 1}),
    ) as open_dcd, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 6, "nsavc": 2},
            io,
            overlap=cfg,
        )

    open_dcd.assert_called_once()
    assert dcd_calls == [1, None]


def test_effective_overlap_check_interval_divides_nstep():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        effective_overlap_check_interval,
    )

    assert effective_overlap_check_interval(2000, 50) == 50
    assert effective_overlap_check_interval(1640, 50) == 41
    assert effective_overlap_check_interval(1650, 50) == 50
    assert effective_overlap_check_interval(100, 50) == 50
    assert effective_overlap_check_interval(7, 50) == 7


def test_effective_overlap_check_interval_respects_nsavc():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        effective_overlap_check_interval,
    )

    assert effective_overlap_check_interval(8000, 50, nsavc=100) == 100
    assert effective_overlap_check_interval(8000, 500, nsavc=100) == 500
    assert effective_overlap_check_interval(8000, 200, nsavc=100) == 200


def test_run_dynamics_with_io_uses_even_overlap_chunks(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=50,
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
    calls: list[int] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(int(kw["nstep"]))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 1640},
            None,
            overlap=cfg,
            overlap_context="HEAT",
        )

    assert all(n == 41 for n in calls)
    assert sum(calls) == 1640
    assert len(calls) == 40


def test_harmonize_dynamics_frequency_for_remainder_chunk():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _harmonize_dynamics_frequency,
        _harmonize_nsavc_frequency,
        _harmonize_overlap_chunk_frequencies,
    )

    assert _harmonize_dynamics_frequency(50, 40) == 40
    assert _harmonize_dynamics_frequency(500, 40) == 40
    assert _harmonize_dynamics_frequency(50, 50) == 50
    assert _harmonize_dynamics_frequency(25, 50) == 25

    assert _harmonize_nsavc_frequency(100, 41) == 1
    assert _harmonize_nsavc_frequency(40, 40) == 20
    assert _harmonize_nsavc_frequency(10, 40) == 10

    kw = {"nstep": 40, "ihbfrq": 50, "imgfrq": 50, "isvfrq": 500, "nsavc": 10, "inbfrq": -1}
    _harmonize_overlap_chunk_frequencies(kw, 40)
    assert kw["ihbfrq"] == 40
    assert kw["imgfrq"] == 40
    assert kw["isvfrq"] == 40
    assert kw["nsavc"] == 10
    assert kw["inbfrq"] == -1

    kw2 = {"nsavc": 10}
    _harmonize_overlap_chunk_frequencies(kw2, 41)
    assert kw2["nsavc"] == 10

    kw3 = {"nsavc": 40}
    _harmonize_overlap_chunk_frequencies(kw3, 40)
    assert kw3["nsavc"] == 40


def test_sync_dynamics_io_units_keeps_explicit_iunrea_minus_one():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _sync_dynamics_io_units

    kw = {"iunrea": -1, "nstep": 50, "restart": False}
    _sync_dynamics_io_units(kw, {"iunwri": 2, "iuncrd": 1})
    assert kw["iunrea"] == -1
    assert "iunwri" not in kw


def test_run_dynamics_chunk_keeps_iunrea_minus_one_for_dynamics():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _run_dynamics_chunk,
    )

    captured: list[dict] = []

    def fake_run(kw):
        captured.append(dict(kw))
        return None

    io = CharmmTrajectoryFiles(restart_write=__import__("pathlib").Path("/tmp/out.res"))
    with mock.patch.object(
        io,
        "open_for_run",
        return_value=([], {"iunwri": 2}),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics",
        side_effect=fake_run,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_charmm_dynamics_rng",
    ):
        _run_dynamics_chunk(
            {"nstep": 50, "restart": False, "iunrea": -1},
            io,
        )

    assert captured[0]["iunrea"] == -1


def test_ensure_nsavc_below_nstep_clamps_full_run():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _ensure_nsavc_below_nstep

    kw = {"nstep": 50, "nsavc": 50}
    _ensure_nsavc_below_nstep(kw)
    assert kw["nsavc"] == 25


def test_resolve_dcd_nsavc_strictly_below_nstep():
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import resolve_dcd_nsavc

    assert resolve_dcd_nsavc(dcd_nsavc=100, nstep=50) == 49
    assert resolve_dcd_nsavc(dcd_nsavc=10, nstep=50) == 10
    assert resolve_dcd_nsavc(dcd_nsavc=5, nstep=2) == 1


def test_overlap_reseeds_rng_before_each_chunk():
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
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_charmm_dynamics_rng",
    ) as refresh, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics",
        return_value=mock.Mock(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 6},
            None,
            overlap=cfg,
            overlap_context="HEAT",
            rng_base=123,
        )

    assert refresh.call_count == 3
    salts = [int(c.kwargs["salt"]) for c in refresh.call_args_list]
    assert len(set(salts)) == 3
    assert all(c.kwargs["base"] == 123 for c in refresh.call_args_list)


def test_refresh_charmm_dynamics_rng_uses_salt_with_base():
    import sys

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _refresh_charmm_dynamics_rng,
    )

    fake_dyn = mock.Mock()
    fake_dyn.get_nrand.return_value = 2
    fake_pycharmm = mock.Mock()
    fake_pycharmm.dynamics = fake_dyn
    with mock.patch.dict(
        sys.modules,
        {"pycharmm": fake_pycharmm, "pycharmm.dynamics": fake_dyn},
    ):
        _refresh_charmm_dynamics_rng(base=123, salt=0)
        _refresh_charmm_dynamics_rng(base=123, salt=1)
    assert fake_dyn.set_rngseeds.call_count == 2
    first = list(fake_dyn.set_rngseeds.call_args_list[0][0][0])
    second = list(fake_dyn.set_rngseeds.call_args_list[1][0][0])
    assert first != second


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
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_charmm_dynamics_rng",
    ), mock.patch(
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
