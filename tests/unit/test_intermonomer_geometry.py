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


def test_contact_summary_marks_tight_prep_contact_for_dcm_like_pair():
    from mmml.utils.intermonomer_geometry import IntermonomerContactSummary

    summary = IntermonomerContactSummary(
        distance_A=1.45,
        threshold_A=1.0,
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
    assert "tight for dynamics" in line
    assert "H" in line and "Cl" in line


def test_contact_summary_ok_when_above_dynamics_guard():
    from mmml.utils.intermonomer_geometry import IntermonomerContactSummary

    summary = IntermonomerContactSummary(
        distance_A=1.561,
        threshold_A=1.0,
        monomer_i=12,
        monomer_j=48,
        atom_i=120,
        atom_j=480,
        label_i="H",
        label_j="Cl",
        dynamics_reference_A=1.5,
    )
    line = summary.format_log_line()
    assert "OK: above dynamics guard" in line


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
        threshold_A=1.0,
        atomic_numbers=[1, 1, 17, 17],
    )
    assert summary.distance_A == pytest.approx(0.45)
    assert summary.label_i in ("H", "Cl")
    assert summary.label_j in ("H", "Cl")
