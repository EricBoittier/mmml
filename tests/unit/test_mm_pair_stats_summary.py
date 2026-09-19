"""format_mm_pair_update_stats_summary helper."""

from mmml.interfaces.pycharmmInterface.mm_energy_forces import format_mm_pair_update_stats_summary


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
