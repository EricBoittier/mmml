"""Unit tests for pre-MLpot inter-monomer distance thresholds."""

from __future__ import annotations

import argparse

import numpy as np
import pytest


def test_resolve_pre_mlpot_ignores_dynamics_overlap_default():
    from mmml.utils.intermonomer_geometry import (
        DEFAULT_PRE_MLPOT_OVERLAP_MIN_A,
        resolve_pre_mlpot_overlap_min_distance,
    )

    args = argparse.Namespace(
        pre_mlpot_overlap_min_distance=None,
        min_intermonomer_atom_distance=0.1,
        dynamics_overlap_min_distance=1.5,
    )
    assert resolve_pre_mlpot_overlap_min_distance(args) == DEFAULT_PRE_MLPOT_OVERLAP_MIN_A


def test_resolve_pre_mlpot_explicit_override():
    from mmml.utils.intermonomer_geometry import resolve_pre_mlpot_overlap_min_distance

    args = argparse.Namespace(
        pre_mlpot_overlap_min_distance=0.8,
        dynamics_overlap_min_distance=1.5,
        min_intermonomer_atom_distance=0.1,
    )
    assert resolve_pre_mlpot_overlap_min_distance(args) == 0.8


def test_resolve_overlap_last_chance_uses_ml_safe_h_heavy_floor():
    from mmml.utils.intermonomer_geometry import (
        DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A,
        resolve_overlap_last_chance_separation_A,
    )

    args = argparse.Namespace(
        pre_mlpot_overlap_min_distance=None,
        min_intermonomer_atom_distance=0.1,
        dynamics_overlap_min_distance=1.5,
        solvents=[],
        composition=None,
        _cluster_composition_summary=None,
    )
    assert resolve_overlap_last_chance_separation_A(args) == pytest.approx(
        DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A
    )

    args_high_prep = argparse.Namespace(
        pre_mlpot_overlap_min_distance=2.8,
        min_intermonomer_atom_distance=0.1,
        dynamics_overlap_min_distance=1.5,
        solvents=[],
        composition=None,
        _cluster_composition_summary=None,
    )
    assert resolve_overlap_last_chance_separation_A(args_high_prep) == pytest.approx(2.8)


def test_resolve_mc_min_uses_prep_floor_under_liquid_prep():
    from mmml.utils.intermonomer_geometry import (
        DEFAULT_PRE_MLPOT_OVERLAP_MIN_A,
        resolve_mc_min_intermonomer_distance_A,
    )

    args = argparse.Namespace(
        liquid_prep=True,
        density_prep_mode=None,
        pre_mlpot_overlap_min_distance=None,
        min_intermonomer_atom_distance=0.1,
    )
    assert resolve_mc_min_intermonomer_distance_A(args) == DEFAULT_PRE_MLPOT_OVERLAP_MIN_A


def test_resolve_mc_min_keeps_packmol_floor_without_liquid_prep():
    from mmml.utils.intermonomer_geometry import resolve_mc_min_intermonomer_distance_A

    args = argparse.Namespace(
        liquid_prep=False,
        density_prep_mode=None,
        min_intermonomer_atom_distance=0.1,
    )
    assert resolve_mc_min_intermonomer_distance_A(args) == pytest.approx(0.1)


def test_dcm_pair_floors_for_h_heavy_and_heavy_heavy():
    from mmml.utils.intermonomer_geometry import (
        DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A,
        DEFAULT_PRE_MLPOT_HEAVY_HEAVY_MIN_A,
        resolve_pre_mlpot_element_pair_min_distance,
    )

    args = argparse.Namespace(solvents=["DCM"])
    assert resolve_pre_mlpot_element_pair_min_distance("H", "Cl", args=args) == pytest.approx(
        DEFAULT_PRE_MLPOT_H_HEAVY_MIN_A
    )
    assert resolve_pre_mlpot_element_pair_min_distance("C", "Cl", args=args) == pytest.approx(
        DEFAULT_PRE_MLPOT_HEAVY_HEAVY_MIN_A
    )


def test_assert_pre_mlpot_mic_geometry_aborts_tight_dcm_contact():
    from mmml.utils.intermonomer_geometry import assert_pre_mlpot_mic_geometry

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    args = argparse.Namespace(
        solvents=["DCM"],
        pre_mlpot_overlap_min_distance=2.3,
        pre_mlpot_h_heavy_min_distance=None,
        pre_mlpot_heavy_heavy_min_distance=None,
        pre_mlpot_h_h_min_distance=2.3,
    )
    with pytest.raises(RuntimeError, match="MIC distance"):
        assert_pre_mlpot_mic_geometry(
            pos,
            [2, 2],
            box_side=None,
            use_pbc=False,
            args=args,
            atomic_numbers=[1, 17, 1, 17],
            context="test",
        )


def test_contact_summary_marks_tight_prep_contact_for_dcm_like_pair():
    from mmml.utils.intermonomer_geometry import IntermonomerContactSummary

    summary = IntermonomerContactSummary(
        distance_A=1.45,
        threshold_A=2.3,
        monomer_i=12,
        monomer_j=48,
        atom_i=120,
        atom_j=480,
        label_i="H",
        label_j="Cl",
        dynamics_reference_A=1.5,
    )
    line = summary.format_log_line()
    assert "1.450" in line
    assert "FAIL: below prep MIC floor" in line
    assert "H" in line and "Cl" in line


def test_contact_summary_ok_when_above_prep_floor():
    from mmml.utils.intermonomer_geometry import IntermonomerContactSummary

    summary = IntermonomerContactSummary(
        distance_A=2.55,
        threshold_A=2.3,
        monomer_i=12,
        monomer_j=48,
        atom_i=120,
        atom_j=480,
        label_i="H",
        label_j="Cl",
        dynamics_reference_A=1.5,
    )
    line = summary.format_log_line()
    assert "OK: above dynamics guard" in line or "passes prep MIC floor" in line


def test_summarize_worst_intermonomer_contact_reports_pair():
    from mmml.utils.intermonomer_geometry import summarize_worst_intermonomer_contact

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.55, 0.0, 0.0],
        ],
        dtype=float,
    )
    summary = summarize_worst_intermonomer_contact(
        pos,
        [2, 2],
        box_side=None,
        use_pbc=False,
        threshold_A=2.3,
        atomic_numbers=[1, 1, 17, 17],
    )
    assert summary.distance_A == pytest.approx(0.45)
    assert summary.label_i in ("H", "Cl")
    assert summary.label_j in ("H", "Cl")
