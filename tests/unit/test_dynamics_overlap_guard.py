"""Tests for PyCHARMM MLpot dynamics overlap guard."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
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
    resolve_overlap_memory_handoff,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics_with_io


@pytest.fixture(autouse=True)
def _mock_bond_exclusion_pairs_unless_targeted(request):
    """Overlap intra checks must not import PyCHARMM in CI."""
    if request.node.name == "test_bond_exclusion_pairs_handles_empty_get_ib_jb":
        yield
        return
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._bond_exclusion_pairs",
        return_value=frozenset(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_restart_write_after_chunk",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._ensure_domdec_off_for_mlpot_energy",
    ):
        yield


def test_bond_exclusion_pairs_handles_empty_get_ib_jb():
    """PyCHARMM can return ``[]`` (not ``([], [])``) when ``get_nbond()==0``."""
    import mmml.interfaces.pycharmmInterface.mlpot.overlap_guard as overlap_guard

    overlap_guard._bond_exclusion_cache = None
    fake_psf = mock.MagicMock()
    fake_psf.get_nbond.return_value = 0
    fake_psf.get_ib_jb.return_value = []
    fake_pycharmm = mock.MagicMock()
    fake_pycharmm.psf = fake_psf
    fake_import = mock.MagicMock()
    with mock.patch.dict(
        sys.modules,
        {
            "pycharmm": fake_pycharmm,
            "pycharmm.psf": fake_psf,
            "mmml.interfaces.pycharmmInterface.import_pycharmm": fake_import,
        },
    ):
        pairs = overlap_guard._bond_exclusion_pairs(exclude_1_3=True)
    assert pairs == frozenset()


def test_monomer_offsets_uniform():
    off = monomer_offsets(10, 2)
    np.testing.assert_array_equal(off, [0, 5, 10])


def test_resolve_defaults_to_rescue_and_1p5A():
    args = argparse.Namespace()
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=True)
    assert cfg.action == "rescue"
    assert cfg.min_distance_A == 1.5
    assert cfg.intra_min_distance_A == 1.0
    assert cfg.intra_exclude_1_3 is True
    assert cfg.check_interval == 500
    assert cfg.enabled is True
    assert cfg.intra_enabled is True
    assert cfg.extent_enabled is True
    assert cfg.max_monomer_extent_A == pytest.approx(12.0)
    assert cfg.rescue.nstep_sd == 200
    assert cfg.rescue.nstep_abnr == 400
    assert cfg.memory_handoff is False


def test_resolve_overlap_memory_handoff_explicit_and_mpi_default(monkeypatch):
    args = argparse.Namespace(dynamics_overlap_memory_handoff=True)
    assert resolve_overlap_memory_handoff(args) is True

    args = argparse.Namespace()
    monkeypatch.delenv("MMML_NO_OVERLAP_MEMORY_HANDOFF", raising=False)
    monkeypatch.delenv("MMML_OVERLAP_MEMORY_HANDOFF", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        assert resolve_overlap_memory_handoff(args) is True
        cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=False)
        assert cfg.memory_handoff is True

    monkeypatch.setenv("MMML_NO_OVERLAP_MEMORY_HANDOFF", "1")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        assert resolve_overlap_memory_handoff(args) is False


def test_resolve_off_disables_inter_and_intra():
    args = argparse.Namespace(dynamics_overlap_action="off")
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=False)
    assert cfg.enabled is False
    assert cfg.intra_enabled is False
    assert cfg.extent_enabled is False


def test_resolve_no_max_extent_disables_extent_guard():
    args = argparse.Namespace(no_dynamics_max_monomer_extent=True)
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=False)
    assert cfg.extent_enabled is False
    assert cfg.max_monomer_extent_A == 0.0


def test_infer_prior_restart_from_write_path(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        attach_prior_segment_restart,
        infer_prior_restart_from_write_path,
    )

    prior = tmp_path / "heat_dcm_90.5.res"
    current = tmp_path / "heat_dcm_90.6.res"
    prior.write_text("prior\n")
    current.write_text("current\n")
    assert infer_prior_restart_from_write_path(current) == prior.resolve()
    cfg = attach_prior_segment_restart(
        DynamicsOverlapConfig(action="rescue", n_monomers=2),
        restart_write=current,
    )
    assert cfg is not None
    assert cfg.prior_segment_restart == prior.resolve()


def test_resolve_intra_min_distance_zero_disables_intra_only():
    args = argparse.Namespace(dynamics_intra_min_distance=0.0)
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=False)
    assert cfg.enabled is True
    assert cfg.intra_enabled is False


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
            intra_min_distance_A=0.0,
            max_monomer_extent_A=0.0,
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
            dmin, _ = check_dynamics_overlap(cfg, context="test", step=0)
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


def test_check_extent_rescue_restores_prior_restart(tmp_path):
    prior = tmp_path / "prior.res"
    prior.write_text("prior restart\n")
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.0,
        intra_min_distance_A=0.0,
        max_monomer_extent_A=12.0,
        n_monomers=2,
        use_pbc=False,
        prior_segment_restart=prior,
        rescue=OverlapRescueConfig(nstep_sd=10, nstep_abnr=0, verbose=False),
    )
    bad_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    good_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    ctx = mock.MagicMock()
    positions = {"current": bad_pos.copy()}

    def _get_pos():
        return positions["current"]

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        side_effect=_get_pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.run_extent_recovery_from_prior_restart",
        side_effect=lambda *a, **k: positions.update(current=good_pos.copy()),
    ) as extent_recovery:
        extent, rescued = check_dynamics_overlap(
            cfg, context="heat segment 2/2", step=1000, mlpot_ctx=ctx
        )
    extent_recovery.assert_called_once()
    assert rescued is True
    assert extent == pytest.approx(np.sqrt(2.0))


def test_check_intra_monomer_raises_on_close_contact():
    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=1.5,
        intra_min_distance_A=1.0,
        n_monomers=1,
        use_pbc=False,
    )
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=float,
    )
    excluded = frozenset({(0, 1), (1, 2)})
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._bond_exclusion_pairs",
        return_value=excluded,
    ):
        with pytest.raises(RuntimeError, match="intra-monomer close contact"):
            check_dynamics_overlap(cfg, context="test", step=100)


def test_check_intra_monomer_rescue_runs_bonded_mini():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.0,
        intra_min_distance_A=1.0,
        n_monomers=1,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=25, nstep_abnr=0, verbose=False),
    )
    pos_bad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ],
        dtype=float,
    )
    pos_ok = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [0.36, 1.03, 0.0],
        ],
        dtype=float,
    )
    excluded = frozenset({(0, 1), (1, 2)})
    ctx = object()
    positions = {"current": pos_bad}

    def _get_pos():
        return positions["current"]

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        side_effect=_get_pos,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._bond_exclusion_pairs",
        return_value=excluded,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._run_intramonomer_bonded_rescue",
        side_effect=lambda _ctx, _cfg: positions.update(current=pos_ok),
    ) as rescue:
        dmin, rescued = check_dynamics_overlap(cfg, context="heat", step=500, mlpot_ctx=ctx)
        rescue.assert_called_once()
    assert rescued
    assert dmin >= 1.0


def test_check_overlap_rescue_runs_minimize_and_rechecks():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        intra_min_distance_A=0.0,
        max_monomer_extent_A=0.0,
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
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.run_inter_monomer_overlap_rescue",
    ) as rescue:
        dmin, rescued = check_dynamics_overlap(
            cfg, context="test", step=50, mlpot_ctx=ctx
        )
        rescue.assert_called_once_with(ctx, cfg)
    assert rescued
    assert dmin > 1.5


def test_check_overlap_rescue_applies_repack_last_resort():
    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=1.5,
        intra_min_distance_A=0.0,
        max_monomer_extent_A=0.0,
        n_monomers=2,
        use_pbc=False,
        rescue=OverlapRescueConfig(nstep_sd=10, nstep_abnr=0, verbose=False),
        separate_on_rescue_fail=True,
        separate_margin_A=0.0,
        repack_spacing_A=4.0,
        recovery_seed=11,
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
        dmin, rescued = check_dynamics_overlap(
            cfg, context="test", step=50, mlpot_ctx=ctx
        )
        rescue.assert_called_once_with(ctx, cfg.rescue)
        sync_pos.assert_called_once()
    assert rescued
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


def test_overlap_checks_run_after_each_successful_chunk(tmp_path):
    """Mid-chunk overlap geometry checks must run at every chunk boundary."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=False,
    )
    overlap_calls: list[int] = []
    chunk_count = 0

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        nonlocal chunk_count
        chunk_count += 1
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text(
                f"REST {500 * chunk_count} 1\n", encoding="utf-8"
            )
        return mock.Mock()

    def track_overlap(_cfg, *, context, step, mlpot_ctx=None):
        overlap_calls.append(int(step))
        return 5.0, False

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        side_effect=track_overlap,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        side_effect=lambda path: int(Path(path).read_text().split()[1]),
    ):
        run_dynamics_with_io(
            {"nstep": 2000},
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat segment 1/8",
        )

    assert chunk_count == 4
    post_chunk_steps = [s for s in overlap_calls if s > 0]
    assert post_chunk_steps == [500, 1000, 1500, 2000]


def test_overlap_checks_run_when_restart_reports_segment_nstep_only(tmp_path):
    """JHSTRT=0 restarts report NSTEP=500 each chunk; checks must still run."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=False,
    )
    overlap_calls: list[int] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST 500 1\n", encoding="utf-8")
        return mock.Mock()

    def track_overlap(_cfg, *, context, step, mlpot_ctx=None):
        overlap_calls.append(int(step))
        return 5.0, False

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        side_effect=track_overlap,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        return_value=500,
    ):
        run_dynamics_with_io(
            {"nstep": 2000},
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat segment 1/8",
        )

    post_chunk_steps = [s for s in overlap_calls if s > 0]
    assert post_chunk_steps == [500, 1000, 1500, 2000]


def test_overlap_skips_check_when_chunk_aborts_early(tmp_path, capsys):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=False,
    )

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST 300 1\n", encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
    ) as check_overlap, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        return_value=300,
    ):
        run_dynamics_with_io(
            {"nstep": 500},
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat segment 1/8",
        )

    post_chunk_steps = [
        call.kwargs["step"]
        for call in check_overlap.call_args_list
        if int(call.kwargs.get("step", -1)) > 0
    ]
    assert post_chunk_steps == []
    assert "echeck or CHARMM abort likely" in capsys.readouterr().out


def test_overlap_post_rescue_handoff_continues_in_memory(tmp_path, capsys):
    """PSF reload during rescue must not READYN stale CPT state on the next chunk."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.5,
        check_interval=500,
        n_monomers=2,
        use_pbc=False,
    )
    calls: list[tuple[dict, object]] = []
    mlpot_ctx = mock.Mock()

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append((dict(kw), _io))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST 500 0\n", encoding="utf-8")
        return mock.Mock()

    def track_overlap(_cfg, *, context, step, mlpot_ctx=None):
        return (5.0, int(step) == 500)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        side_effect=track_overlap,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_restart_write_after_chunk",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.patch_restart_global_step",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_post_rescue_overlap_handoff",
    ) as post_rescue, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        side_effect=lambda path: 500 * len(calls),
    ):
        run_dynamics_with_io(
            {"nstep": 1500},
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat segment 1/8",
            mlpot_ctx=mlpot_ctx,
        )

    assert len(calls) == 3
    assert calls[0][0]["restart"] is False
    assert calls[1][0]["restart"] is False
    assert calls[1][1] is not None and calls[1][1].restart_read is None
    assert calls[2][0]["restart"] is False
    assert calls[2][1] is not None and calls[2][1].restart_read is None
    post_rescue.assert_called_once()
    out = capsys.readouterr().out
    assert "post-rescue in-memory handoff at global step 500" in out
    assert "remaining chunks" in out


def test_overlap_post_rescue_single_chunk_patches_without_extra_dyna(tmp_path, capsys):
    """Segment-boundary rescue with one overlap chunk must not require another dyna leg."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="rescue",
        min_distance_A=0.5,
        check_interval=2500,
        n_monomers=2,
        use_pbc=False,
    )
    calls: list[dict] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text(
                "REST    48     0\n"
                "\n"
                " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
                "          25         500         500         500          10           0\n",
                encoding="utf-8",
            )
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        side_effect=lambda *_a, **kw: (5.0, int(kw.get("step", 0)) == 2500),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_restart_write_after_chunk",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.patch_restart_global_step",
        return_value=True,
    ) as patch_step, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_post_rescue_overlap_handoff",
    ) as post_rescue, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_segment_restart_after_overlap_rescue",
    ) as refresh_segment:
        run_dynamics_with_io(
            {"nstep": 2500},
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat segment 1/8",
        )

    assert len(calls) == 1
    post_rescue.assert_not_called()
    refresh_segment.assert_called_once()
    patch_step.assert_called()
    out = capsys.readouterr().out
    assert "post-rescue restart refreshed" in out
    assert "segment complete; no extra dyna" in out
    assert "in-memory handoff" not in out


def test_mlpot_overlap_chunks_continue_in_memory_without_readyn(tmp_path):
    """Legacy --dynamics-overlap-memory-handoff skips READYN between chunks."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=2,
        n_monomers=2,
        use_pbc=False,
        memory_handoff=True,
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
    res_path = tmp_path / "heat.res"
    io = CharmmTrajectoryFiles(restart_write=res_path)
    calls: list[dict] = []
    mlpot_ctx = mock.Mock()

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text("REST 48 -1\n", encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        return_value=(5.0, False),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        return_value=2,
    ):
        run_dynamics_with_io(
            {"nstep": 6, "new": False, "start": False, "restart": False},
            io,
            overlap=cfg,
            overlap_context="HEAT",
            mlpot_ctx=mlpot_ctx,
        )

    assert [c["nstep"] for c in calls] == [2, 2, 2]
    assert calls[0]["restart"] is False
    assert calls[1]["restart"] is False
    assert calls[1].get("iunrea") == -1
    assert calls[1]["iasvel"] == 0
    assert calls[2]["restart"] is False
    assert calls[2]["iasvel"] == 0


def test_mlpot_overlap_chunks_use_scratch_restart_handoff(tmp_path):
    """Default MLpot overlap uses dyna restart on alternating scratch .res files."""
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
    res_path = tmp_path / "heat.res"
    io = CharmmTrajectoryFiles(restart_write=res_path)
    calls: list[dict] = []
    mlpot_ctx = mock.Mock()
    valid_restart = (
        "REST     1     500\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "   10     0       2       1      10     500     297       0       0\n"
    )

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text(valid_restart, encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        return_value=(5.0, False),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_restart_write_after_chunk",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        return_value=2,
    ):
        run_dynamics_with_io(
            {"nstep": 6, "new": False, "start": False, "restart": False},
            io,
            overlap=cfg,
            overlap_context="HEAT",
            mlpot_ctx=mlpot_ctx,
        )

    assert [c["nstep"] for c in calls] == [2, 2, 2]
    assert calls[0]["restart"] is False
    assert calls[1]["restart"] is True
    assert calls[1]["iasvel"] == 0
    assert calls[2]["restart"] is True
    assert calls[2]["iasvel"] == 0


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


def test_overlap_first_chunk_drops_restart_when_restart_read_is_invalid(tmp_path):
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
    heat = tmp_path / "heat.res"
    heat.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    io = CharmmTrajectoryFiles(
        restart_read=heat,
        restart_write=tmp_path / "equi.res",
    )
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
                "nstep": 4,
                "new": False,
                "start": False,
                "restart": True,
                "iunrea": 3,
            },
            io,
            overlap=cfg,
            overlap_context="equi",
        )

    assert calls[0]["restart"] is False
    assert calls[0]["iunrea"] == -1
    assert calls[1]["restart"] is True


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

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
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
    slot_a.write_text("REST\n", encoding="utf-8")
    slot_b.write_text("REST\n", encoding="utf-8")

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


def test_overlap_chunk_io_skips_nonrestartable_scratch_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_chunk_io,
        _overlap_restart_slot_paths,
    )

    heat = tmp_path / "heat.res"
    equi = tmp_path / "equi_dcm_90.res"
    heat.write_text("REST\n", encoding="utf-8")
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    slot_a, _ = _overlap_restart_slot_paths(equi)
    slot_a.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )

    c1 = _overlap_chunk_io(io, chunk_index=1, n_chunks=3)

    assert c1.restart_read is None
    assert c1.restart_write == equi.with_name("equi_dcm_90.overlap_b.res")


def test_overlap_chunk_io_skips_negative_restart_header(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_chunk_io,
        _overlap_restart_slot_paths,
    )

    heat = tmp_path / "heat.res"
    equi = tmp_path / "equi_dcm_10.res"
    heat.write_text("REST\n", encoding="utf-8")
    io = CharmmTrajectoryFiles(restart_read=heat, restart_write=equi)
    slot_a, _ = _overlap_restart_slot_paths(equi)
    slot_a.write_text("REST    48    -1                \n", encoding="utf-8")

    c1 = _overlap_chunk_io(io, chunk_index=1, n_chunks=3)

    assert c1.restart_read is None
    assert c1.restart_write == equi.with_name("equi_dcm_10.overlap_b.res")


def test_overlap_aborts_before_charmm_when_scratch_restart_is_invalid(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_restart_slot_paths,
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
    res_path = tmp_path / "heat.res"
    io = CharmmTrajectoryFiles(restart_write=res_path)
    slot_a, _ = _overlap_restart_slot_paths(res_path)
    calls: list[dict] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        slot_a.write_text(
            "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
            encoding="utf-8",
        )
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_overlap_scratch_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        with pytest.raises(RuntimeError, match="scratch restart.*is not restartable after chunk 1"):
            run_dynamics_with_io(
                {"nstep": 6, "nsavc": 1},
                io,
                overlap=cfg,
                overlap_context="heat",
            )

    assert len(calls) == 1


def test_assign_velocities_at_temperature_uses_nstep_zero():
    from unittest.mock import patch

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        assign_velocities_at_temperature,
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics"
    ) as run_dyn:
        assign_velocities_at_temperature(48.0, use_pbc=False)
    kw = run_dyn.call_args.args[0]
    assert int(kw["nstep"]) == 0
    assert int(kw["iasvel"]) == 1
    assert float(kw["firstt"]) == 48.0


def test_overlap_config_for_stage_heat_uses_single_segment():
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
        overlap_config_for_stage,
    )

    cfg = DynamicsOverlapConfig(
        action="rescue",
        check_interval=500,
        n_monomers=90,
    )
    heat_cfg = overlap_config_for_stage(cfg, stage="heat", nstep=4000)
    assert heat_cfg is not None
    assert int(heat_cfg.check_interval) == 4000
    staged_heat = overlap_config_for_stage(
        cfg, stage="heat", nstep=1000, n_segments=4
    )
    assert staged_heat is not None
    assert int(staged_heat.check_interval) == 1000
    nve_cfg = overlap_config_for_stage(cfg, stage="nve", nstep=8000)
    assert nve_cfg is not None
    assert int(nve_cfg.check_interval) == 500


def test_overlap_should_split_trajectory_limits_chunk_dcd_count():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _overlap_should_split_trajectory,
    )

    assert not _overlap_should_split_trajectory(n_chunks=1, traj_nsavc=1)
    assert not _overlap_should_split_trajectory(n_chunks=7390, traj_nsavc=1)
    assert not _overlap_should_split_trajectory(n_chunks=100, traj_nsavc=1)
    assert not _overlap_should_split_trajectory(n_chunks=9, traj_nsavc=1)
    assert _overlap_should_split_trajectory(n_chunks=50, traj_nsavc=100)
    assert _overlap_should_split_trajectory(n_chunks=4, traj_nsavc=100)
    assert _overlap_should_split_trajectory(n_chunks=8, traj_nsavc=1)
    assert _overlap_should_split_trajectory(n_chunks=3, traj_nsavc=100)


def test_overlap_chunk_continues_velocity_scaling_heat_ramp(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=500,
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
    calls: list[dict] = []
    mlpot_ctx = mock.Mock()
    valid_restart = (
        "REST     1     500\n"
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL\n"
        "   10     0     500       1      10    2000     297       0       0\n"
    )

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        calls.append(dict(kw))
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text(valid_restart, encoding="utf-8")
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard.check_dynamics_overlap",
        return_value=(5.0, False),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_restart_write_after_chunk",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_last_step",
        side_effect=lambda path: 500 * len(calls),
    ):
        run_dynamics_with_io(
            {
                "nstep": 2000,
                "firstt": 0.0,
                "finalt": 240.0,
                "ihtfrq": 100,
                "TEMINC": 0.12,
                "iasors": 0,
            },
            CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res"),
            overlap=cfg,
            overlap_context="heat",
            mlpot_ctx=mlpot_ctx,
        )

    assert len(calls) == 4
    assert calls[0]["firstt"] == 0.0
    assert calls[1]["restart"] is True
    assert calls[1]["iasvel"] == 0
    assert calls[1]["firstt"] == pytest.approx(0.6)
    assert calls[2]["firstt"] == pytest.approx(1.2)
    assert calls[3]["firstt"] == pytest.approx(1.8)


def test_apply_overlap_chunk_hoover_chunk0_omits_start_and_iasvel_one():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _apply_overlap_chunk_dynamics_kw

    kw = {
        "start": False,
        "firstt": 0.0,
        "finalt": 240.0,
        "cpt": True,
        "hoover reft": 240.0,
        "iasvel": 0,
    }
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=0, has_restart_read=False)
    assert kw["start"] is False
    assert kw["iasvel"] == 1
    assert kw["restart"] is False


def test_apply_overlap_chunk_preserves_heat_cold_start_kw():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import _apply_overlap_chunk_dynamics_kw

    kw = {
        "start": False,
        "firstt": 48.0,
        "finalt": 240.0,
        "iasors": 0,
        "iasvel": 1,
        "ihtfrq": 1000,
        "TEMINC": 48.0,
    }
    _apply_overlap_chunk_dynamics_kw(kw, chunk_index=0, has_restart_read=False)
    assert kw["start"] is False
    assert kw["firstt"] == 48.0
    assert kw["finalt"] == 240.0
    assert kw["iasors"] == 0
    assert kw["iasvel"] == 1
    assert kw["ihtfrq"] == 1000
    assert kw["TEMINC"] == 48.0
    assert kw["iunrea"] == -1
    assert kw["restart"] is False

    kw2 = {"start": False, "firstt": 48.0, "iasvel": 1}
    _apply_overlap_chunk_dynamics_kw(kw2, chunk_index=0, has_restart_read=False)
    assert kw2["firstt"] == 48.0
    assert kw2["iasvel"] == 1

    kw3 = {
        "start": False,
        "firstt": 48.0,
        "finalt": 240.0,
        "tbath": 240.0,
        "iasors": 0,
        "iasvel": 1,
        "ihtfrq": 1000,
        "TEMINC": 48.0,
    }
    _apply_overlap_chunk_dynamics_kw(kw3, chunk_index=0, has_restart_read=False)
    assert kw3["firstt"] == 48.0
    assert kw3["finalt"] == 240.0
    assert kw3["iasors"] == 0
    assert kw3["iasvel"] == 1
    assert kw3["ihtfrq"] == 1000
    assert kw3["TEMINC"] == 48.0


def test_ensure_valid_overlap_scratch_restart_raises_on_rest_minus_one(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _ensure_valid_overlap_scratch_restart,
        _overlap_restart_slot_paths,
    )

    final = tmp_path / "heat.res"
    slot_a, _ = _overlap_restart_slot_paths(final)
    slot_a.write_text("REST    48    -1                \n", encoding="utf-8")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_overlap_scratch_restart",
    ):
        with pytest.raises(RuntimeError, match="REST step field='-1'"):
            _ensure_valid_overlap_scratch_restart(
                slot_a,
                final_restart=final,
                chunk_index=0,
                n_chunks=8,
                overlap_context="HEAT",
            )


def test_overlap_refresh_scratch_restart_fixes_invalid_handoff(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_restart_slot_paths,
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
    res_path = tmp_path / "heat.res"
    io = CharmmTrajectoryFiles(restart_write=res_path)
    slot_a, slot_b = _overlap_restart_slot_paths(res_path)
    refreshed: list[Path] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        if _io is not None and _io.restart_write is not None:
            Path(_io.restart_write).write_text(
                "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
                encoding="utf-8",
            )
        return mock.Mock()

    def fake_refresh(write_path, *, final_restart):
        if write_path is not None:
            p = Path(write_path)
            p.write_text("REST\n", encoding="utf-8")
            refreshed.append(p)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._run_dynamics_chunk",
        side_effect=fake_chunk,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._refresh_overlap_scratch_restart",
        side_effect=fake_refresh,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._prepare_overlap_chunk_after_restart",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 6, "nsavc": 1},
            io,
            overlap=cfg,
            overlap_context="heat",
        )

    assert slot_a in refreshed
    assert slot_b in refreshed


def test_overlap_multi_chunk_splits_dcd_per_chunk(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        CharmmTrajectoryFiles,
        _overlap_chunk_trajectory_path,
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
    dcd = tmp_path / "heat.dcd"
    io = CharmmTrajectoryFiles(restart_write=tmp_path / "heat.res", trajectory=dcd)
    chunk_paths: list[Path | None] = []
    expected_chunks = [_overlap_chunk_trajectory_path(dcd, i) for i in range(3)]

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        chunk_paths.append(_io.trajectory if _io is not None else None)
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
    ), mock.patch(
        "mmml.utils.dcd_writer.concat_dcd_files",
    ) as merge:
        run_dynamics_with_io(
            {"nstep": 6, "nsavc": 1},
            io,
            overlap=cfg,
            overlap_context="heat",
        )

    assert chunk_paths == expected_chunks
    merge.assert_not_called()


def test_overlap_single_chunk_uses_stage_dcd_directly(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles

    cfg = DynamicsOverlapConfig(
        action="error",
        min_distance_A=0.5,
        check_interval=10,
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
    chunk_paths: list[Path | None] = []

    def fake_chunk(kw, _io, *, extra_iokw=None, **kwargs):
        chunk_paths.append(_io.trajectory if _io is not None else None)
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
            {"nstep": 6, "nsavc": 2},
            io,
            overlap=cfg,
        )

    assert chunk_paths == [dcd]


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
    assert kw2["nsavc"] == 1

    kw3 = {"nsavc": 40}
    _harmonize_overlap_chunk_frequencies(kw3, 40)
    assert kw3["nsavc"] == 20


def test_apply_dyn_imgfrq_from_args_sets_pbc_list_freqs():
    from argparse import Namespace

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        apply_dyn_imgfrq_from_args,
    )

    kw = {"imgfrq": 50, "ihbfrq": 50, "ilbfrq": 50}
    apply_dyn_imgfrq_from_args(kw, Namespace(dyn_imgfrq=400), charmm_pbc=True)
    assert kw["imgfrq"] == 400
    assert kw["ihbfrq"] == 400
    assert kw["ilbfrq"] == 400

    kw_vac = {"imgfrq": 50}
    apply_dyn_imgfrq_from_args(kw_vac, Namespace(dyn_imgfrq=400), charmm_pbc=False)
    assert kw_vac["imgfrq"] == 50


def test_apply_loose_pbc_dyn_freq_kwargs_above_nstep():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        apply_loose_pbc_dyn_freq_kwargs,
    )

    kw = {
        "nstep": 1250,
        "ntrfrq": 1000,
        "ixtfrq": 1000,
        "imgfrq": 50,
        "ihbfrq": 50,
        "ilbfrq": 50,
        "inbfrq": -1,
    }
    apply_loose_pbc_dyn_freq_kwargs(kw, nstep=1250)
    assert kw["ntrfrq"] == 1251
    assert kw["ixtfrq"] == 1251
    assert kw["imgfrq"] == 1251
    assert kw["ihbfrq"] == 1251
    assert kw["ilbfrq"] == 1251
    assert kw["inbfrq"] == -1


def test_ensure_ntrfrq_above_nstep_for_non_loose_pbc():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _ensure_ntrfrq_above_nstep,
    )

    kw = {"ntrfrq": 1000, "nstep": 1250}
    _ensure_ntrfrq_above_nstep(kw, 1250)
    assert kw["ntrfrq"] == 1251

    kw2 = {"ntrfrq": 1000, "nstep": 40000}
    _ensure_ntrfrq_above_nstep(kw2, 40000)
    assert kw2["ntrfrq"] == 40001

    kw3 = {"ntrfrq": 2000, "nstep": 1250}
    _ensure_ntrfrq_above_nstep(kw3, 1250)
    assert kw3["ntrfrq"] == 2000

    kw4 = {"ntrfrq": 0, "nstep": 1250}
    _ensure_ntrfrq_above_nstep(kw4, 1250)
    assert kw4["ntrfrq"] == 0


def test_harmonize_overlap_chunk_non_loose_pbc_lifts_ntrfrq():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _harmonize_overlap_chunk_frequencies,
    )

    kw = {"ntrfrq": 1000, "nstep": 1250, "iprfrq": 1250, "isvfrq": 1250}
    _harmonize_overlap_chunk_frequencies(kw, 1250, loose_pbc=False)
    assert kw["ntrfrq"] == 1251


def test_harmonize_overlap_chunk_loose_pbc_disables_image_freqs():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _harmonize_overlap_chunk_frequencies,
    )

    kw = {
        "nstep": 40,
        "ntrfrq": 1000,
        "ihbfrq": 50,
        "imgfrq": 50,
        "ixtfrq": 1000,
        "ilbfrq": 50,
        "isvfrq": 500,
        "nsavc": 10,
        "inbfrq": -1,
    }
    _harmonize_overlap_chunk_frequencies(kw, 40, loose_pbc=True)
    assert kw["ntrfrq"] == 41
    assert kw["ihbfrq"] == 41
    assert kw["imgfrq"] == 41
    assert kw["ixtfrq"] == 41
    assert kw["ilbfrq"] == 41
    assert kw["isvfrq"] == 40
    assert kw["inbfrq"] == -1


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
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=nullcontext(),
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
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=nullcontext(),
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
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        return_value=nullcontext(),
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


def test_prepare_overlap_chunk_skips_upinb_when_mlpot_active():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _prepare_overlap_chunk_after_restart,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext

    ctx = mock.Mock(spec=MlpotContext)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._import_pycharmm_modules",
    ) as imp:
        _prepare_overlap_chunk_after_restart(ctx)
    imp.assert_not_called()


def test_prepare_post_rescue_overlap_handoff_assigns_velocities_in_memory():
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        _prepare_post_rescue_overlap_handoff,
    )

    chunk_kw = {
        "firstt": 40.4,
        "tbath": 63.0,
        "timestep": 0.0001,
        "cpt": True,
        "restart": True,
        "iunrea": 3,
    }
    ctx = mock.Mock(use_pbc=True, charmm_cubic_box_side_A=180.0)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.assign_velocities_at_temperature",
    ) as assign_vel, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.pbc_env.ensure_charmm_crystal_for_cpt",
    ) as ensure_crystal:
        _prepare_post_rescue_overlap_handoff(chunk_kw, mlpot_ctx=ctx)

    assign_vel.assert_called_once_with(
        63.0,
        timestep_ps=0.0001,
        restart_path=None,
        use_pbc=True,
    )
    ensure_crystal.assert_called_once_with(180.0, quiet=True)
    assert chunk_kw["restart"] is False
    assert chunk_kw["iasvel"] == 0
    assert chunk_kw["iunrea"] == -1
    assert "finalt" not in chunk_kw
