"""Tests for PyCHARMM MLpot dynamics overlap guard."""

from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
    DynamicsOverlapConfig,
    check_dynamics_overlap,
    monomer_offsets,
    resolve_dynamics_overlap_config,
)
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import run_dynamics_with_io


def test_monomer_offsets_uniform():
    off = monomer_offsets(10, 2)
    np.testing.assert_array_equal(off, [0, 5, 10])


def test_resolve_defaults_to_error_and_1p5A():
    args = argparse.Namespace()
    cfg = resolve_dynamics_overlap_config(args, n_monomers=4, use_pbc=True)
    assert cfg.action == "error"
    assert cfg.min_distance_A == 1.5
    assert cfg.check_interval == 50
    assert cfg.enabled is True


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


def test_run_dynamics_with_io_chunks_and_checks():
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
    calls: list[dict] = []

    def fake_run(kw):
        calls.append(dict(kw))
        return mock.Mock()

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.run_dynamics",
        side_effect=fake_run,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos_ok,
    ):
        run_dynamics_with_io(
            {"nstep": 5, "new": True, "start": True, "restart": True, "iunrea": 3},
            overlap=cfg,
            overlap_context="NVE",
        )

    assert [c["nstep"] for c in calls] == [2, 2, 1]
    assert calls[0]["restart"] is True
    assert calls[0]["iunrea"] == 3
    assert calls[1]["restart"] is False
    assert calls[1]["iunrea"] == -1
    assert calls[2]["restart"] is False
    assert calls[2]["iunrea"] == -1
