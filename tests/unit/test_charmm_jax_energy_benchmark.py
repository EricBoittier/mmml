"""Unit tests for CHARMM vs JAX energy benchmark reporting (no PyCHARMM)."""

from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
    ForceDelta,
    LayerBenchmark,
    SUPPORTED_CASES,
    SystemBenchmark,
    TermDelta,
    _relative_diff,
    all_layers_passed,
    render_json_report,
    render_markdown_report,
)


def test_relative_diff():
    assert _relative_diff(100.0, 100.1) == pytest.approx(0.001)
    assert _relative_diff(0.0, 0.01) == pytest.approx(0.01 / 1e-12)


def test_render_markdown_and_json_smoke():
    layer = LayerBenchmark(
        layer="bonded",
        n_atoms=3,
        terms=(
            TermDelta.from_pair("bond", 1.0, 1.0001),
            TermDelta.from_pair("total", 2.5, 2.5002),
        ),
        forces=ForceDelta(force_rms=0.001, force_max=0.01),
        passed=True,
    )
    case = SystemBenchmark(
        name="tip3_monomer",
        description="smoke",
        n_atoms=3,
        layers=(layer,),
    )
    md = render_markdown_report((case,))
    assert "tip3_monomer" in md
    assert "PASS" in md
    assert "| bond |" in md

    js = render_json_report((case,))
    assert '"tip3_monomer"' in js
    assert '"layers_passed": 1' in js


def test_all_layers_passed():
    ok = LayerBenchmark("bonded", 1, (), None, True)
    bad = LayerBenchmark("bonded", 1, (), None, False)
    case_ok = SystemBenchmark("a", "", 1, (ok,))
    case_bad = SystemBenchmark("b", "", 1, (bad,))
    assert all_layers_passed((case_ok,))
    assert not all_layers_passed((case_bad,))


def test_supported_cases():
    assert "tip3_monomer" in SUPPORTED_CASES
    assert "trialanine_water" in SUPPORTED_CASES
