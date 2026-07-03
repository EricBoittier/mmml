"""Regression tests for CHARMM DCD staging / formatted-unformatted I/O crashes.

Aborted ``mini_box_equil`` runs can leave partial binary DCDs under
``$TMPDIR/mmml-charmm-io``.  Reopening those aliases via ``dynamics_set_iuncrd``
without clearing them triggers ``Format present for UNFORMATTED data transfer``
in ``dynio.F90`` (READYN on a trajectory unit).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from mmml.interfaces.pycharmmInterface import charmm_paths
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles


def _staging_alias(
    target: Path,
    staging: Path,
    *,
    payload: bytes = b"partial-dcd",
) -> Path:
    alias_obj = charmm_paths.charmm_io_alias(
        target,
        for_write=True,
        staging_root=staging,
    )
    assert alias_obj is not None
    alias_obj.alias.write_bytes(payload)
    return alias_obj.alias


def test_reset_stage_trajectory_removes_output_and_staging(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    staging = tmp_path / "staging"
    monkeypatch.setenv("MMML_CHARMM_IO_STAGING", str(staging))
    dcd = tmp_path / "pretreat" / "mini_box_equil.dcd"
    dcd.parent.mkdir(parents=True)
    dcd.write_bytes(b"old-output-dcd")
    alias = _staging_alias(dcd, staging, payload=b"stale-staging-dcd")

    _reset_stage_trajectory(dcd)

    assert not dcd.is_file()
    assert not alias.is_file()


def test_hot_and_cold_mini_box_equil_staging_aliases_are_independent(
    tmp_path,
    monkeypatch,
):
    staging = tmp_path / "staging"
    monkeypatch.setenv("MMML_CHARMM_IO_STAGING", str(staging))
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    hot = pretreat / "mini_box_equil_hot.dcd"
    cold = pretreat / "mini_box_equil.dcd"

    hot_alias = _staging_alias(hot, staging, payload=b"hot-partial")
    cold_alias = _staging_alias(cold, staging, payload=b"cold-partial")
    assert hot_alias != cold_alias

    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    _reset_stage_trajectory(cold)

    assert hot_alias.is_file()
    assert hot_alias.read_bytes() == b"hot-partial"
    assert not cold_alias.is_file()


def test_open_for_run_after_reset_has_no_stale_staging_alias(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    monkeypatch.setenv("MMML_CHARMM_IO_STAGING", str(staging))
    dcd = tmp_path / "pretreat" / "mini_box_equil.dcd"
    dcd.parent.mkdir(parents=True)
    stale_alias = _staging_alias(dcd, staging)

    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    _reset_stage_trajectory(dcd)
    assert not stale_alias.is_file()

    io = CharmmTrajectoryFiles(restart_write=dcd.with_suffix(".res"), trajectory=dcd)
    _open_files, iokw, aliases = io.open_for_run()

    assert _open_files == []
    assert isinstance(iokw["iuncrd"], str)
    assert "mini_box_equil.dcd" in iokw["iuncrd"]
    assert aliases
    alias_path = Path(iokw["iuncrd"])
    assert not alias_path.is_file() or alias_path.stat().st_size == 0


def test_charmm_io_alias_append_preserves_existing_staging(tmp_path):
    target = tmp_path / "pretreat" / "mini_box_equil.dcd"
    staging = tmp_path / "staging"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"committed-dcd")

    alias = _staging_alias(target, staging, payload=b"append-me")
    append_alias = charmm_paths.charmm_io_alias(
        target,
        for_write=True,
        append=True,
        staging_root=staging,
    )
    assert append_alias is not None
    assert append_alias.alias.is_file()
    assert append_alias.alias.read_bytes() == b"append-me"


def test_mini_box_equil_heat_leg_resets_trajectory_before_dynamics(tmp_path):
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    paths = {
        "mini_box_equil_res": pretreat / "mini_box_equil.res",
        "mini_box_equil_dcd": pretreat / "mini_box_equil.dcd",
        "mini_box_equil_hot_res": pretreat / "mini_box_equil_hot.res",
        "mini_box_equil_hot_dcd": pretreat / "mini_box_equil_hot.dcd",
    }
    args = argparse.Namespace(
        quiet=True,
        save=True,
        no_echeck=False,
        no_echeck_heat=False,
        rescue_old_dcd=False,
        seed=1,
        dcd_nsavc=None,
    )
    call_order: list[str] = []

    def _track_reset(path, **kwargs):
        call_order.append(f"reset:{Path(path).name}")

    def _fake_dynamics(*_a, **_k):
        call_order.append("dynamics")
        from types import SimpleNamespace

        return SimpleNamespace()

    with (
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow._reset_stage_trajectory",
            side_effect=_track_reset,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics_with_io",
            side_effect=_fake_dynamics,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.assert_stage_dynamics_completed",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.apply_pretreat_dyn_freq_kwargs",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_dcd_nsavc",
            return_value=100,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_heat_ihtfrq",
            return_value=100,
        ),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
            _run_mini_box_equil_heat_leg,
        )

        _run_mini_box_equil_heat_leg(
            args,
            paths=paths,
            res_key="mini_box_equil_res",
            dcd_key="mini_box_equil_dcd",
            timestep_ps=0.00025,
            duration_ps=0.1,
            firstt=20.0,
            finalt=120.0,
            echeck=500.0,
            use_pbc=True,
            coords_in_memory=True,
            restart_read_key=None,
            overlap_context="MINI_BOX_EQUIL_COLD",
        )

    assert call_order == ["reset:mini_box_equil.dcd", "dynamics"]


def test_mini_box_equil_cold_leg_resets_only_cold_dcd(tmp_path):
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    paths = {
        "mini_box_equil_res": pretreat / "mini_box_equil.res",
        "mini_box_equil_dcd": pretreat / "mini_box_equil.dcd",
        "mini_box_equil_hot_res": pretreat / "mini_box_equil_hot.res",
        "mini_box_equil_hot_dcd": pretreat / "mini_box_equil_hot.dcd",
    }
    args = argparse.Namespace(
        quiet=True,
        save=True,
        no_echeck=False,
        no_echeck_heat=False,
        rescue_old_dcd=False,
        seed=1,
        dcd_nsavc=None,
    )
    reset_paths: list[str] = []

    def _capture_reset(path, **kwargs):
        reset_paths.append(Path(path).name)

    with (
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.staged_workflow._reset_stage_trajectory",
            side_effect=_capture_reset,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics_with_io",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.assert_stage_dynamics_completed",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.apply_pretreat_dyn_freq_kwargs",
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_dcd_nsavc",
            return_value=100,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_heat_ihtfrq",
            return_value=100,
        ),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.box_equil import (
            _run_mini_box_equil_heat_leg,
        )

        _run_mini_box_equil_heat_leg(
            args,
            paths=paths,
            res_key="mini_box_equil_hot_res",
            dcd_key="mini_box_equil_hot_dcd",
            timestep_ps=0.00025,
            duration_ps=0.1,
            firstt=20.0,
            finalt=120.0,
            echeck=500.0,
            use_pbc=True,
            coords_in_memory=True,
            restart_read_key=None,
            overlap_context="MINI_BOX_EQUIL_HOT",
        )

    assert reset_paths == ["mini_box_equil_hot.dcd"]


@pytest.mark.parametrize(
    "rescue_old",
    [False, True],
)
def test_reset_stage_trajectory_always_clears_staging_even_with_rescue_old(
    tmp_path,
    monkeypatch,
    rescue_old: bool,
):
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    staging = tmp_path / "staging"
    monkeypatch.setenv("MMML_CHARMM_IO_STAGING", str(staging))
    dcd = tmp_path / "pretreat" / "mini_box_equil.dcd"
    dcd.parent.mkdir(parents=True)
    dcd.write_bytes(b"output-dcd")
    alias = _staging_alias(dcd, staging)

    _reset_stage_trajectory(dcd, rescue_old=rescue_old)

    assert not alias.is_file()
    if rescue_old:
        assert not dcd.is_file()
        rescued = list(dcd.parent.glob("mini_box_equil.rescued.*.dcd"))
        assert len(rescued) == 1
    else:
        assert not dcd.is_file()
