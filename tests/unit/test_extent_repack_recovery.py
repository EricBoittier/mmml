"""Extent repack recovery: in-memory references and FIRE polish."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.extent_repack_recovery import (
    resolve_extent_reference_positions,
    stash_geometry_reference_on_ctx,
)
from mmml.interfaces.pycharmmInterface.mlpot.setup import MlpotContext


def test_stash_geometry_reference_on_ctx_mini():
    ctx = MlpotContext(
        mlpot=mock.MagicMock(),
        pyCModel=mock.MagicMock(),
        params=None,
        model=None,
    )
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
        return_value=pos,
    ):
        stash_geometry_reference_on_ctx(ctx, kind="mini")
    assert ctx.geometry_mini_positions is not None
    np.testing.assert_allclose(ctx.geometry_mini_positions, pos)


def test_resolve_extent_reference_falls_back_to_memory_mini():
    ref = np.array(
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
    ctx = MlpotContext(
        mlpot=mock.MagicMock(),
        pyCModel=mock.MagicMock(),
        params=None,
        model=None,
        geometry_mini_positions=ref.copy(),
    )
    arr, path = resolve_extent_reference_positions([], ctx)
    np.testing.assert_allclose(arr, ref)
    assert "in-memory-mini" in str(path)


def test_resolve_extent_reference_prefers_disk_over_memory(tmp_path):
    crd = tmp_path / "02_mini.crd"
    disk_ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    mem = np.zeros((6, 3), dtype=float)
    ctx = MlpotContext(
        mlpot=mock.MagicMock(),
        pyCModel=mock.MagicMock(),
        params=None,
        model=None,
        geometry_mini_positions=mem,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._load_extent_reference_positions",
        return_value=(disk_ref, crd.resolve()),
    ):
        arr, path = resolve_extent_reference_positions([crd], ctx)
    assert path == crd.resolve()
    assert float(arr[1, 0]) == pytest.approx(2.0)
