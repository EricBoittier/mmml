"""Neighbor-list summary: sparse-dimer capacity fill bar at setup time."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from mmml.cli.run.summaries import print_neighbor_list_summary


def test_print_neighbor_list_summary_shows_sparse_dimer_fill_bar() -> None:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100, color_system=None)
    print_neighbor_list_summary(
        n_atoms=1644,
        n_monomers=548,
        cell_L_A=31.868,
        mm_cutoff_A=11.0,
        capacity_multiplier=1.75,
        skin_distance_A=0.25,
        update_interval_steps=1,
        extra={
            "ml_sparse_dimers": True,
            "dimers_total": 149878,
            "max_active_dimers": 11482,
            "ml_batch_size": "all",
            "ml_gpu_count": 1,
            "PBC": True,
        },
        console=console,
    )
    text = buf.getvalue()
    assert "Capacity fill-fraction" in text
    assert "Sparse ML dimers" in text
    assert "11482" in text.replace(",", "")
    assert "149878" in text.replace(",", "")
