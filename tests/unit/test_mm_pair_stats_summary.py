"""format_mm_pair_update_stats_summary helper."""

import numpy as np

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    _mm_pair_stats_init,
    format_mm_pair_update_stats_summary,
    resolve_mm_pair_stats_n_valid,
)


def test_empty_stats():
    line = format_mm_pair_update_stats_summary({})
    assert "0/0 reused" in line


def test_occupancy_and_radius_in_summary():
    line = format_mm_pair_update_stats_summary(
        {
            "calls": 10,
            "reused": 8,
            "pair_capacity": 3600000,
            "pair_n_valid": 660000,
            "radius": {
                "list_radius_A": 12.99,
                "interaction_radius_A": 12.75,
                "skin_A": 0.24,
                "box_half_min_A": 13.0,
            },
        }
    )
    assert "occupancy=660000/3600000 (18.3%)" in line
    assert "list=12.990 A" in line
    assert "interaction 12.750 + skin 0.240" in line
    assert "L/2=13.000" in line


def test_resolve_mm_pair_stats_n_valid_prefers_explicit_count():
    assert resolve_mm_pair_stats_n_valid(n_valid=7, pair_mask=np.ones(3)) == 7


def test_resolve_mm_pair_stats_n_valid_from_pair_mask():
    """jax-md allocate never binds ``_n_valid``; occupancy comes from the mask."""
    assert resolve_mm_pair_stats_n_valid(pair_mask=np.array([1.0, 0.0, 1.0, 1.0])) == 3
    assert resolve_mm_pair_stats_n_valid() is None


def test_mm_pair_stats_init_seeds_occupancy_from_mask():
    stats = _mm_pair_stats_init(
        n_static_pairs=100,
        n_valid=resolve_mm_pair_stats_n_valid(pair_mask=np.array([1, 1, 0, 0])),
        radius_info={},
        update_interval=1,
        skin_distance=0.25,
        capacity_multiplier=1.25,
    )
    assert stats["pair_n_valid"] == 2
    assert "occupancy=2/100" in format_mm_pair_update_stats_summary(stats)
