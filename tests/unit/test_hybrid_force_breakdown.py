"""Unit tests for hybrid ModelOutput force-term breakdown."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mmml.analysis.hybrid_force_breakdown import (
    FORCE_TERM_RESIDUAL_NOISE_EVA,
    force_magnitude_stats,
    hybrid_force_term_breakdown,
)


def test_force_magnitude_stats_basic():
    f = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    s = force_magnitude_stats(f, name="t", energy_eV=-1.5)
    assert s.name == "t"
    assert s.max_abs_eVA == pytest.approx(5.0)
    assert s.mean_abs_eVA == pytest.approx(2.5)
    assert s.energy_eV == pytest.approx(-1.5)


def test_breakdown_identifies_dominant_mm():
    n = 4
    internal = np.zeros((n, 3))
    ml_2b = np.zeros((n, 3))
    mm = np.zeros((n, 3))
    mm[:, 0] = 6.0
    total = internal + ml_2b + mm
    out = SimpleNamespace(
        forces=total,
        energy=-100.0,
        internal_E=10.0,
        internal_F=internal,
        ml_2b_E=-20.0,
        ml_2b_F=ml_2b,
        mm_E=-90.0,
        mm_F=mm,
        mm_vdw_E=-30.0,
        mm_elec_E=-60.0,
        wall_E=0.0,
        mbd_E=0.0,
    )
    br = hybrid_force_term_breakdown(out, atomic_numbers=np.array([8, 1, 1, 8]))
    assert br["dominant_term_by_mean_abs_F"] == "mm"
    terms = {t["name"]: t for t in br["terms"]}
    assert terms["mm"]["mean_abs_eVA"] == pytest.approx(6.0)
    assert terms["internal_ML"]["mean_abs_eVA"] == pytest.approx(0.0)
    assert terms["residual_wall_mbd_restraints"]["max_abs_eVA"] < (
        10 * FORCE_TERM_RESIDUAL_NOISE_EVA
    )
    assert "8" in br["by_element_total_F"]


def test_breakdown_residual_captures_wall_like_force():
    n = 2
    internal = np.zeros((n, 3))
    ml_2b = np.zeros((n, 3))
    mm = np.zeros((n, 3))
    wall = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    total = wall
    out = SimpleNamespace(
        forces=total,
        energy=1.0,
        internal_E=0.0,
        internal_F=internal,
        ml_2b_E=0.0,
        ml_2b_F=ml_2b,
        mm_E=0.0,
        mm_F=mm,
        mm_vdw_E=0.0,
        mm_elec_E=0.0,
        wall_E=1.0,
        mbd_E=0.0,
    )
    br = hybrid_force_term_breakdown(out)
    assert br["dominant_term_by_mean_abs_F"] == "residual_wall_mbd_restraints"
    terms = {t["name"]: t for t in br["terms"]}
    assert terms["residual_wall_mbd_restraints"]["max_abs_eVA"] == pytest.approx(2.0)
