"""Tests for sparse ML dimer cap policy."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    SparseDimerCapOverflow,
    max_dimer_pairs,
    raise_if_sparse_cap_saturated,
    resolve_max_active_dimers,
    sparse_dimer_active_radius,
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


@pytest.mark.parametrize("box,near", [(None, 2), (10.0, 3)])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_validate_sparse_dimer_cap_counts_and_reports_overflow(box, near, cap):
    # Distances: 1, 8.5, 7.5 in free space; 1, 1.5, 2.5 under MIC.
    pos = np.array([[0., 0., 0.], [1., 0., 0.], [8.5, 0., 0.]])
    stats = validate_sparse_dimer_cap(
        pos, 3, 1, mm_switch_on=8.0, box_side_A=box, max_active_dimers=cap,
    )
    assert stats["n_dimers_total"] == 3
    assert stats["n_near_mm_switch_on"] == near
    assert stats["cap_margin"] == cap - near
    assert stats["cap_saturated"] is (near > cap)
    assert stats["ok"] is (near <= cap)
    assert stats["physnet_systems_per_step"] == 3 + min(near, cap)
    assert stats["verdict"].startswith("FAIL:" if near > cap else "WARN:")


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


def test_resolve_max_active_dimers_density_aware_covers_tip3_548_liquid():
    """TIP3:548 @ ~32 Å had ~10780 in-range pairs vs ~8200 uniform estimate.

    The 1.4 safety margin must clear that undercount without an explicit cap.
    """
    n_monomers = 548
    box_side_A = 31.86819225525291
    box_volume = box_side_A**3
    active_radius = 6.0 + 1.5
    n_dimers_total = max_dimer_pairs(n_monomers)
    real_measured_near_pairs = 10786

    cap = resolve_max_active_dimers(
        n_monomers, n_dimers_total, box_volume=box_volume, active_radius=active_radius
    )
    assert cap >= real_measured_near_pairs
    # Old 1.1 margin gave ~9020 and saturated; keep a clear gap above measured.
    assert cap > 11000


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


def test_sparse_dimer_active_radius_is_mm_switch_on():
    """Outer switch support ends at mm_switch_on; extra width is not extra radius."""
    assert sparse_dimer_active_radius(6.0, 1.5) == 6.0
    assert sparse_dimer_active_radius(8.0, 1.5) == 8.0
    assert sparse_dimer_active_radius(6.0, 1.5, margin=1.5) == 7.5
    with pytest.raises(ValueError, match="margin"):
        sparse_dimer_active_radius(6.0, margin=-0.1)


def test_raise_if_sparse_cap_saturated():
    raise_if_sparse_cap_saturated(-1, 10)
    raise_if_sparse_cap_saturated(10, 10)
    with pytest.raises(SparseDimerCapOverflow) as exc:
        raise_if_sparse_cap_saturated(11, 10)
    assert "would be dropped" in str(exc.value)


def test_resolve_max_active_dimers_without_density_info_unchanged():
    """Backward compatibility: omitting box_volume/active_radius must give
    the exact pre-fix result (existing callers that don't pass them yet)."""
    n_monomers = 903
    n_dimers_total = max_dimer_pairs(n_monomers)
    assert resolve_max_active_dimers(n_monomers, n_dimers_total) == 5418

