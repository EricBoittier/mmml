"""Unit tests for optional CHARMM nonbond debug snapshots."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_nbond_debug_enabled_env(monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_nbond_diagnostics import (
        nbond_debug_enabled,
    )

    monkeypatch.delenv("MMML_NBOND_DEBUG", raising=False)
    monkeypatch.delenv("MMML_SAVE_NBOND_SNAPSHOTS", raising=False)
    assert nbond_debug_enabled() is False
    monkeypatch.setenv("MMML_NBOND_DEBUG", "1")
    assert nbond_debug_enabled() is True


def test_maybe_snapshot_nbond_state_writes_json(tmp_path, monkeypatch):
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_nbond_diagnostics import (
        maybe_snapshot_nbond_state,
    )

    monkeypatch.setenv("MMML_NBOND_DEBUG_DIR", str(tmp_path))
    ctx = MagicMock()
    ctx.use_pbc = True
    ctx.cubic_box_side_A = 30.0
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.charmm_nbond_diagnostics.collect_nbond_state",
        return_value={"context": "unit", "timestamp_unix": 1.0, "natom": 3},
    ):
        path = maybe_snapshot_nbond_state(ctx, context="unit", force=True)
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["natom"] == 3
