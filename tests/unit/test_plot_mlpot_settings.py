"""Tests for MLpot settings / cutoff documentation plots."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

NEW_FIGURES = (
    "docs/images/mlpot-settings/cutoff_radius_ladder.png",
    "docs/images/mlpot-settings/system_monomer_regions.png",
    "docs/images/mlpot-settings/dual_stack_responsibilities.png",
    "docs/images/mlpot-settings/lr_solvers_overview.png",
    "docs/images/mlpot-settings/lr_energy_split_schematic.png",
)


def test_plot_mlpot_settings_generates_figures():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "plot_mlpot_settings.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    for rel in NEW_FIGURES:
        path = REPO / rel
        assert path.is_file(), f"missing plot: {rel}"
        assert path.stat().st_size > 500
