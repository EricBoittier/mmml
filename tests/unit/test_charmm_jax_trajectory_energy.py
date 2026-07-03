"""Unit tests for trajectory PyCHARMM vs JAX energy reporting (no PyCHARMM)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.charmm_jax_trajectory_energy import (
    FrameEnergyComparison,
    TermTrajectoryStats,
    TrajectoryEnergyComparison,
    render_trajectory_json,
    render_trajectory_markdown,
    summarize_trajectory_term_errors,
    synthetic_trajectory_from_seed,
    term_deltas_from_component_maps,
)
from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
    ForceDelta,
    TermDelta,
)


def test_term_deltas_from_component_maps():
    jax = {"bond": 1.0, "total": 3.0}
    charmm = {"bond": 1.1, "total": 3.2}
    deltas = term_deltas_from_component_maps(jax, charmm, term_order=("bond", "total"))
    assert len(deltas) == 2
    assert deltas[0].term == "bond"
    assert deltas[0].abs_diff == pytest.approx(-0.1)
    assert deltas[1].abs_diff == pytest.approx(-0.2)


def test_summarize_trajectory_term_errors():
    frame_a = FrameEnergyComparison(
        frame=0,
        terms=(
            TermDelta.from_pair("bond", 1.0, 1.01),
            TermDelta.from_pair("total", 10.0, 10.05),
        ),
        forces=ForceDelta(force_rms=0.01, force_max=0.05),
    )
    frame_b = FrameEnergyComparison(
        frame=1,
        terms=(
            TermDelta.from_pair("bond", 1.0, 1.03),
            TermDelta.from_pair("total", 10.0, 10.10),
        ),
        forces=ForceDelta(force_rms=0.02, force_max=0.08),
    )
    stats = summarize_trajectory_term_errors((frame_a, frame_b))
    bond = next(s for s in stats if s.term == "bond")
    assert bond.max_abs_diff == pytest.approx(0.03)
    assert bond.mean_abs_diff == pytest.approx(0.02)
    assert bond.rms_abs_diff == pytest.approx((0.01**2 + 0.03**2) ** 0.5 / (2**0.5))


def test_render_trajectory_reports_smoke():
    frame = FrameEnergyComparison(
        frame=0,
        terms=(TermDelta.from_pair("vdw", -2.0, -2.001),),
        forces=ForceDelta(force_rms=0.001, force_max=0.01),
    )
    report = TrajectoryEnergyComparison(
        name="tip3_water_box",
        description="smoke",
        n_atoms=30,
        n_frames=1,
        frames=(frame,),
        term_stats=(
            TermTrajectoryStats(
                term="vdw",
                max_abs_diff=0.001,
                mean_abs_diff=0.001,
                rms_abs_diff=0.001,
                max_rel_diff=5e-4,
                mean_rel_diff=5e-4,
            ),
        ),
    )
    md = render_trajectory_markdown(report)
    assert "tip3_water_box" in md
    assert "Frame 0" in md
    assert "| vdw |" in md

    js = render_trajectory_json(report)
    assert '"tip3_water_box"' in js
    assert '"max_abs_diff": 0.001' in js


def test_synthetic_trajectory_shape_and_determinism():
    pos0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    a = synthetic_trajectory_from_seed(pos0, n_frames=4, seed=7)
    b = synthetic_trajectory_from_seed(pos0, n_frames=4, seed=7)
    assert a.shape == (4, 2, 3)
    assert a[0, 0, 0] == pytest.approx(0.0)
    assert (a == b).all()
    assert not np.allclose(a[0], a[1])
