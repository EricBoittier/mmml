"""Unit tests for liquid-density diagnostic helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "pbc_liquid_density_dyn"
sys.path.insert(0, str(WORKFLOW / "scripts"))

import monitor_lib as ml  # noqa: E402
import trajectory_diag as td  # noqa: E402


def test_parse_dyna_lines():
    text = """
DYNA>      100      0.0250   -1234.5    300.0    -1534.5    298.2
DYNA>      200      0.0500   -1230.1    305.0    -1535.1    299.0
"""
    rows = ml.parse_dyna_lines(text)
    assert len(rows) == 2
    assert rows[0]["step"] == 100.0
    assert rows[1]["temperature_K"] == pytest.approx(299.0)


def test_summarize_dyna_drift():
    rows = [
        {"step": 1, "time_ps": 0.0, "total_energy_kcal": 0.0, "kinetic_energy_kcal": 0.0,
         "potential_energy_kcal": -10.0, "temperature_K": 300.0},
        {"step": 2, "time_ps": 0.1, "total_energy_kcal": 1.5, "kinetic_energy_kcal": 0.0,
         "potential_energy_kcal": -9.0, "temperature_K": 301.0},
    ]
    summary = ml.summarize_dyna(rows)
    assert summary["n_frames"] == 2
    assert summary["total_energy_drift_kcal"] == pytest.approx(1.5)


def test_compute_rdf_g():
    frames = [np.array([[0, 0, 0], [3.0, 0, 0]], dtype=float)]
    rdf = td.compute_rdf_g(frames, r_max=6.0, n_bins=30)
    assert rdf["n_frames"] == 1
    assert len(rdf["g_r"]) == 30
    assert rdf["peak_r_A"] is not None
