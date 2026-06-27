"""Unit tests for per-monomer hybrid force diagnostics."""

from __future__ import annotations

import numpy as np
import pytest


def test_per_monomer_forces_grms_single_atom_monomer():
    from mmml.utils.monomer_force_diag import per_monomer_forces_grms_kcalmol_A

    forces = np.array([[3.0, 4.0, 0.0], [0.0, 3.0, 4.0]], dtype=float)
    offsets = np.array([0, 1, 2], dtype=int)
    per_mono = per_monomer_forces_grms_kcalmol_A(forces, offsets)
    assert per_mono.shape == (2,)
    assert per_mono[0] == pytest.approx(np.sqrt(25.0 / 3.0))
    assert per_mono[1] == pytest.approx(np.sqrt(25.0 / 3.0))


def test_select_stressed_monomer_indices_flags_one_hot_spot():
    from mmml.utils.monomer_force_diag import select_stressed_monomer_indices

    grms = np.array([5.0, 5.0, 120.0, 4.0, 6.0], dtype=float)
    flagged = select_stressed_monomer_indices(grms, max_select=2, min_abs_grms=30.0)
    assert flagged == [2]


def test_select_stressed_monomer_indices_empty_when_widespread():
    from mmml.utils.monomer_force_diag import select_stressed_monomer_indices

    grms = np.array([80.0, 75.0, 90.0, 85.0], dtype=float)
    assert select_stressed_monomer_indices(grms, max_select=2) == []


def test_diagnose_monomer_forces_merges_overlap_hint_when_within_max_select():
    from mmml.utils.monomer_force_diag import diagnose_monomer_forces

    forces = np.zeros((4, 3), dtype=float)
    forces[2:4] = 100.0
    offsets = np.array([0, 2, 4], dtype=int)
    diag = diagnose_monomer_forces(
        forces,
        offsets,
        max_select=2,
        min_abs_grms=30.0,
        min_ratio_to_median=2.0,
        overlap_monomers=(0, 1),
    )
    assert sorted(diag.flagged) == [0, 1]


def test_diagnose_monomer_forces_clears_flag_when_merge_exceeds_max_select():
    from mmml.utils.monomer_force_diag import diagnose_monomer_forces

    forces = np.zeros((6, 3), dtype=float)
    forces[4:6] = 100.0
    offsets = np.array([0, 2, 4, 6], dtype=int)
    diag = diagnose_monomer_forces(
        forces,
        offsets,
        max_select=2,
        min_abs_grms=30.0,
        overlap_monomers=(0, 1),
    )
    assert diag.flagged == ()


def test_resolve_selective_repack_monomers_uses_mlpot_ctx(monkeypatch):
    from mmml.utils.monomer_force_diag import resolve_selective_repack_monomers

    offsets = np.array([0, 2, 4], dtype=int)
    forces = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0],
        ],
        dtype=float,
    )

    monkeypatch.setattr(
        "mmml.utils.monomer_force_diag.mlpot_hybrid_forces_kcalmol_A",
        lambda *_a, **_kw: forces,
    )
    ctx = object()
    diag = resolve_selective_repack_monomers(
        ctx, offsets, max_select=2, min_abs_grms=10.0, min_ratio_to_median=1.5
    )
    assert diag is not None
    assert diag.flagged == (1,)
