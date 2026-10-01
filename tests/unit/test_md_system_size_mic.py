"""Headline MD sizes must sit in the unique-MIC regime (L >= 2 × cutoff)."""

from __future__ import annotations

from benchmarks.benchmarks._common import water_box
from benchmarks.benchmarks.bench_md_driver import CUTOFF_A, MDSystemSize


def test_md_system_size_params_are_mic():
    for n_waters in MDSystemSize.params:
        box = water_box(int(n_waters))
        assert box["box_L"] >= 2.0 * CUTOFF_A, (
            f"{n_waters} waters: L={box['box_L']:.3f} Å < 2×{CUTOFF_A} Å"
        )


def test_216_waters_is_below_the_mic_floor():
    box = water_box(216)
    assert box["box_L"] < 2.0 * CUTOFF_A
    assert 216 not in MDSystemSize.params
