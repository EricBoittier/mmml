"""Tests for overlap run-state sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
    load_overlap_run_state,
    save_overlap_run_state,
)


def test_save_and_load_overlap_run_state_round_trip(tmp_path):
    directory = tmp_path / "overlap"
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    save_overlap_run_state(
        directory,
        step=500,
        segment="heat segment 1/8",
        chunk_index=0,
        positions=positions,
        restart_path=tmp_path / "heat.res",
        quiet=True,
    )
    tree = load_overlap_run_state(directory)
    np.testing.assert_allclose(tree["positions"], positions)
    assert tree["metadata"]["step"] == 500
    assert tree["metadata"]["chunk_index"] == 0
