"""format_mm_pair_update_stats_summary helper."""

from mmml.interfaces.pycharmmInterface.mm_energy_forces import format_mm_pair_update_stats_summary


def test_empty_stats():
    line = format_mm_pair_update_stats_summary({})
    assert "0/0 reused" in line
