"""Unit tests for prep-ladder geometry checkpoint diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics import (
    print_geometry_checkpoint_diff,
)


def test_print_geometry_checkpoint_diff_reports_rmsd(capsys):
    before = np.zeros((4, 3), dtype=np.float64)
    after = before.copy()
    after[0, 0] = 0.5
    ctx = MagicMock(topology_psf_path=None)

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics.print_topology_composition_note",
    ):
        print_geometry_checkpoint_diff(
            before,
            after,
            step_label="round1:bonded_mm",
            mlpot_ctx=ctx,
        )

    out = capsys.readouterr().out
    assert "round1:bonded_mm geometry diff" in out
    assert "RMSD=0.2500" in out
    assert "max|Δ|=0.5000" in out
    assert "no cluster PSF for stretch analysis" in out


def test_print_geometry_checkpoint_diff_bond_stretch_delta(capsys, tmp_path):
    before = np.zeros((2, 3), dtype=np.float64)
    before[1, 0] = 1.0
    after = before.copy()
    after[1, 0] = 1.8
    psf = tmp_path / "cluster.psf"
    psf.write_text("psf\n", encoding="utf-8")

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics._bond_stretch_summary",
        side_effect=[
            {"n_bonds": 1, "n_stretched": 0, "max_stretch_A": 0.0, "mean_stretch_A": 0.0},
            {"n_bonds": 1, "n_stretched": 1, "max_stretch_A": 0.8, "mean_stretch_A": 0.8},
        ],
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint_diagnostics.print_topology_composition_note",
    ):
        print_geometry_checkpoint_diff(
            before,
            after,
            step_label="round2:bonded_mm",
            topology_psf=psf,
        )

    out = capsys.readouterr().out
    assert "stretched 0 -> 1" in out
    assert "max stretch 0.000 -> 0.800" in out
