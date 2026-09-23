"""MM pair-list radius arithmetic: interaction + skin must stay strictly below L/2."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    check_mm_pair_list_radius,
    format_mm_pair_list_radius_report,
    mm_pair_list_radius_breakdown,
    resolve_mm_pair_list_cutoff_A,
)


def test_26A_box_12p75_interaction_rejects_0p4_skin() -> None:
    """L=26 Å → L/2=13 Å. Interaction 12.75 Å leaves <0.25 Å for skin, not 0.4 Å."""
    cell = np.eye(3) * 26.0
    # 6.0 + 1.5 + 2*2.625 = 12.75 interaction (quoted "12.75 + skin").
    bad = mm_pair_list_radius_breakdown(
        mm_switch_on=6.0,
        mm_switch_width=1.5,
        skin_distance=0.4,
        assumed_extent_A=2.625,
        cell=cell,
    )
    assert bad["interaction_radius_A"] == pytest.approx(12.75)
    assert bad["list_radius_A"] == pytest.approx(13.15)
    assert bad["box_half_min_A"] == pytest.approx(13.0)
    assert bad["max_legal_skin_A"] == pytest.approx(0.25)
    assert bad["skin_legal"] is False
    with pytest.raises(ValueError, match="reaches half the box"):
        check_mm_pair_list_radius(bad["list_radius_A"], cell, detail=format_mm_pair_list_radius_report(bad))

    still_equal = mm_pair_list_radius_breakdown(
        mm_switch_on=6.0,
        mm_switch_width=1.5,
        skin_distance=0.25,
        assumed_extent_A=2.625,
        cell=cell,
    )
    assert still_equal["list_radius_A"] == pytest.approx(13.0)
    assert still_equal["skin_legal"] is False

    ok = mm_pair_list_radius_breakdown(
        mm_switch_on=6.0,
        mm_switch_width=1.5,
        skin_distance=0.24,
        assumed_extent_A=2.625,
        cell=cell,
    )
    assert ok["list_radius_A"] == pytest.approx(12.99)
    assert ok["skin_legal"] is True
    check_mm_pair_list_radius(ok["list_radius_A"], cell)


def test_report_logs_every_term() -> None:
    info = mm_pair_list_radius_breakdown(
        mm_switch_on=6.0,
        mm_switch_width=1.5,
        skin_distance=0.25,
        measured_extent_A=2.375,
        extent_margin_A=0.25,
        cell=np.eye(3) * 30.0,
    )
    text = format_mm_pair_list_radius_report(info)
    assert "interaction radius" in text
    assert "12.7500" in text or "12.75" in text
    assert "L/2" in text
    assert "max legal skin" in text
    assert "OK" in text
    assert resolve_mm_pair_list_cutoff_A(6.0, 1.5, 0.25, molecule_extent_A=2.625) == pytest.approx(
        info["list_radius_A"]
    )


def test_shared_cutoff_mode_ignores_extent() -> None:
    info = mm_pair_list_radius_breakdown(
        shared_cutoff=8.0,
        skin_distance=0.25,
        assumed_extent_A=99.0,
        cell=np.eye(3) * 20.0,
    )
    assert info["mode"] == "shared_cutoff"
    assert info["interaction_radius_A"] == pytest.approx(8.0)
    assert info["list_radius_A"] == pytest.approx(8.25)
    assert info["twice_assumed_extent_A"] == 0.0
