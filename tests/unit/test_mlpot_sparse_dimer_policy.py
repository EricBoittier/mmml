"""Tests for sparse ML dimer cap policy."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    max_dimer_pairs,
    resolve_max_active_dimers,
    validate_sparse_dimer_cap,
)


def test_resolve_max_active_dimers_default_90():
    assert resolve_max_active_dimers(90, 4005) == 4005


def test_resolve_max_active_dimers_pbc_50_uses_all_pairs():
    assert resolve_max_active_dimers(50, 1225) == 1225


def test_resolve_max_active_dimers_free_space_uses_all_pairs():
    assert max_dimer_pairs(90) == 4005
    assert resolve_max_active_dimers(90, 4005, free_space=True) == 4005


def test_resolve_max_active_dimers_free_space_promotes_lower_explicit():
    assert resolve_max_active_dimers(90, 4005, explicit=1000, free_space=True) == 4005


def test_resolve_max_active_dimers_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_MAX_ACTIVE_DIMERS", "1500")
    assert resolve_max_active_dimers(90, 4005) == 1500


def test_resolve_max_active_dimers_free_space_promotes_lower_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_MAX_ACTIVE_DIMERS", "1500")
    assert resolve_max_active_dimers(90, 4005, free_space=True) == 4005


def test_resolve_max_active_dimers_small_cluster():
    assert resolve_max_active_dimers(5, 10) == 10


def test_validate_sparse_dimer_cap_random_sparse():
    rng = np.random.default_rng(0)
    n = 20
    apm = 10
    pos = rng.standard_normal((n * apm, 3)) * 5.0
    stats = validate_sparse_dimer_cap(pos, n, apm, mm_switch_on=7.0, box_side_A=None)
    assert stats["n_dimers_total"] == n * (n - 1) // 2
    assert "verdict" in stats
    assert isinstance(stats["ok"], bool)


def test_count_near_dimer_pairs_free_space_cap_is_all_pairs():
    n = 10
    apm = 5
    pos = np.zeros((n * apm, 3), dtype=np.float64)
    stats = validate_sparse_dimer_cap(pos, n, apm, mm_switch_on=7.0, free_space=True)
    assert stats["max_active_dimers_cap"] == n * (n - 1) // 2
    assert stats["free_space"] is True


def test_resolve_max_active_dimers_flat_heuristic_undersizes_real_liquid_water():
    """Locks in the bug this module's density-aware branch fixes.

    Numbers are from the TIP3:903, L=30.307409163768842 A NVE run that
    motivated this fix (mmml_calculator.py active_radius =
    mm_switch_on + ml_switch_width = 7.5 A): every recorded frame of the
    actual trajectory had ~26,470-26,493 monomer pairs within that radius,
    while the flat "6 neighbors/monomer" heuristic caps at 5,418 -- a ~79.5%
    silent truncation of in-range ML-dimer pairs every step.
    """
    n_monomers = 903
    n_dimers_total = max_dimer_pairs(n_monomers)
    real_measured_near_pairs = 26480  # actual trajectory, see docstring above

    flat_cap = resolve_max_active_dimers(n_monomers, n_dimers_total)
    assert flat_cap == max(4005, 6 * n_monomers) == 5418
    assert flat_cap < real_measured_near_pairs, (
        "this assertion documents the pre-fix bug: the flat heuristic must "
        "NOT cover the real near-pair count without box density info"
    )


def test_resolve_max_active_dimers_density_aware_covers_real_liquid_water():
    """Same real-run numbers as above, but with box_volume/active_radius
    supplied (the PBC path `mmml_calculator.setup_calculator` now uses) --
    the resulting cap must comfortably cover the actually-measured near-pair
    count from the real trajectory, not just an idealized estimate.
    """
    n_monomers = 903
    box_side_A = 30.307409163768842
    box_volume = box_side_A**3
    active_radius = 6.0 + 1.5  # mm_switch_on + ml_switch_width for this run
    n_dimers_total = max_dimer_pairs(n_monomers)
    real_measured_near_pairs = 26480

    cap = resolve_max_active_dimers(
        n_monomers, n_dimers_total, box_volume=box_volume, active_radius=active_radius
    )
    assert cap > real_measured_near_pairs
    margin = (cap - real_measured_near_pairs) / real_measured_near_pairs
    assert margin > 0.2, f"expected a healthy safety margin, got {margin:.1%}"


def test_resolve_max_active_dimers_density_aware_never_below_flat_fallback():
    """The density-aware branch must not regress a caller relying on the old
    floor for a sparse/dilute PBC system (e.g. a solute in a huge box)."""
    n_monomers = 200
    n_dimers_total = max_dimer_pairs(n_monomers)
    huge_box_volume = 1.0e9  # dilute enough that density-aware estimate -> ~0
    cap = resolve_max_active_dimers(
        n_monomers, n_dimers_total, box_volume=huge_box_volume, active_radius=6.0
    )
    assert cap >= max(4005, 6 * n_monomers)


def test_resolve_max_active_dimers_without_density_info_unchanged():
    """Backward compatibility: omitting box_volume/active_radius must give
    the exact pre-fix result (existing callers that don't pass them yet)."""
    n_monomers = 903
    n_dimers_total = max_dimer_pairs(n_monomers)
    assert resolve_max_active_dimers(n_monomers, n_dimers_total) == 5418
