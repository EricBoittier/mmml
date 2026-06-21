"""Tests for geometry checkpoint ladder and pretreat resume."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
    attempt_overlap_early_abort_recovery,
    build_geometry_recovery_candidates,
    discover_resume_restart,
    first_valid_restart_path,
    is_overlap_scratch_restart_path,
    pretreat_stage_complete,
    resolve_geometry_checkpoint_ladder,
    resume_charmm_mm_pretreat_if_available,
    write_geometry_baseline_restart,
)
from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig


def test_geometry_ladder_prefers_baseline_over_pretreat_prod(tmp_path):
    tag = "dcm_90"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    paths = {
        "heat_res": tmp_path / f"heat_{tag}.res",
        "charmm_mm_prod_res": pretreat / f"charmm_mm_prod_{tag}.res",
        "charmm_mm_equi_res": pretreat / f"charmm_mm_equi_{tag}.res",
        "charmm_mm_heat_res": pretreat / f"charmm_mm_heat_{tag}.res",
        "geometry_baseline_res": tmp_path / f"geometry_baseline_{tag}.res",
    }
    paths["geometry_baseline_res"].write_text("baseline\n", encoding="utf-8")
    paths["charmm_mm_prod_res"].write_text("prod\n", encoding="utf-8")

    ladder = resolve_geometry_checkpoint_ladder(paths, tag, n_heat_segments=1)
    assert ladder.index(paths["geometry_baseline_res"]) < ladder.index(
        paths["charmm_mm_prod_res"]
    )
    found = first_valid_restart_path(ladder)
    assert found == paths["geometry_baseline_res"].resolve()


def test_discover_resume_restart_prefers_heat_segment(tmp_path):
    tag = "dcm_10"
    heat_seg = tmp_path / f"heat_{tag}.1.res"
    heat_seg.write_text("heat seg\n", encoding="utf-8")
    baseline = tmp_path / f"geometry_baseline_{tag}.res"
    baseline.write_text("baseline\n", encoding="utf-8")
    paths = {
        "heat_res": tmp_path / f"heat_{tag}.res",
        "geometry_baseline_res": baseline,
    }
    found = discover_resume_restart(tmp_path, tag, paths=paths, n_heat_segments=2)
    assert found == heat_seg.resolve()


def test_pretreat_resume_skips_completed_heat(tmp_path):
    tag = "dcm_190"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    heat = pretreat / f"charmm_mm_heat_{tag}.res"
    heat.write_text("heat\n", encoding="utf-8")
    paths = {
        "charmm_mm_prod_res": pretreat / f"charmm_mm_prod_{tag}.res",
        "charmm_mm_equi_res": pretreat / f"charmm_mm_equi_{tag}.res",
        "charmm_mm_heat_res": heat,
    }
    args = argparse.Namespace(
        charmm_mm_pretreat_ps_equi=2.0,
        charmm_mm_pretreat_ps_prod=2.0,
        charmm_mm_pretreat_ps_heat=2.0,
        charmm_mm_pretreat_heat_nstep=10000,
        dcd_nsavc=100,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.pretreat_stage_complete",
        side_effect=lambda path, **kw: path == heat,
    ):
        state = resume_charmm_mm_pretreat_if_available(
            paths,
            args,
            timestep_ps=0.0002,
        )
    assert state.skip_heat is True
    assert state.skip_minimize is True
    assert state.restart_read == heat.resolve()


def test_pretreat_resume_skips_completed_legs(tmp_path):
    tag = "dcm_10"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    prod = pretreat / f"charmm_mm_prod_{tag}.res"
    prod.write_text("prod\n", encoding="utf-8")
    paths = {
        "charmm_mm_prod_res": prod,
        "charmm_mm_equi_res": pretreat / f"charmm_mm_equi_{tag}.res",
        "charmm_mm_heat_res": pretreat / f"charmm_mm_heat_{tag}.res",
    }
    args = argparse.Namespace(
        charmm_mm_pretreat_ps_equi=100.0,
        charmm_mm_pretreat_ps_prod=100.0,
        charmm_mm_pretreat_ps_heat=100.0,
        charmm_mm_pretreat_heat_nstep=2000,
        dcd_nsavc=100,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.pretreat_stage_complete",
        side_effect=lambda path, **kw: path == prod,
    ):
        state = resume_charmm_mm_pretreat_if_available(
            paths,
            args,
            timestep_ps=0.0005,
        )
    assert state.skip_entire_pretreat is True
    assert state.restart_read == prod.resolve()


def test_write_geometry_baseline_restart(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
        return_value=True,
    ) as rewrite:
        path = write_geometry_baseline_restart(out, "dcm_10")
    rewrite.assert_called_once()
    assert path == out / "geometry_baseline_dcm_10.res"


def test_pretreat_resume_continues_partial_heat(tmp_path):
    tag = "aco_266"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    heat = pretreat / f"charmm_mm_heat_{tag}.res"
    heat.write_text("heat partial\n", encoding="utf-8")
    paths = {
        "charmm_mm_prod_res": pretreat / f"charmm_mm_prod_{tag}.res",
        "charmm_mm_equi_res": pretreat / f"charmm_mm_equi_{tag}.res",
        "charmm_mm_heat_res": heat,
    }
    args = argparse.Namespace(
        charmm_mm_pretreat_ps_equi=2.0,
        charmm_mm_pretreat_ps_prod=2.0,
        charmm_mm_pretreat_ps_heat=2.0,
        charmm_mm_pretreat_heat_nstep=10000,
        dcd_nsavc=100,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.pretreat_stage_complete",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.first_valid_restart_path",
        return_value=heat.resolve(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.resolve_integrated_restart_step",
        return_value=1800,
    ):
        state = resume_charmm_mm_pretreat_if_available(
            paths,
            args,
            timestep_ps=0.0002,
        )
    assert state.skip_heat is False
    assert state.skip_minimize is True
    assert state.heat_integrated_step == 1800
    assert state.restart_read == heat.resolve()


def test_pretreat_stage_complete_uses_integrated_step(tmp_path):
    res = tmp_path / "heat.res"
    res.write_text("REST\n", encoding="utf-8")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics._valid_restart_file",
        return_value=res.resolve(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.resolve_integrated_restart_step",
        return_value=950,
    ):
        assert pretreat_stage_complete(res, expected_nstep=1000) is True


def test_is_overlap_scratch_restart_path():
    assert is_overlap_scratch_restart_path("heat_dcm_90.0.overlap_a.res")
    assert is_overlap_scratch_restart_path("/tmp/heat.overlap_b.res")
    assert not is_overlap_scratch_restart_path("heat_dcm_90.0.res")
    assert not is_overlap_scratch_restart_path("geometry_baseline_dcm_90.res")


def test_build_geometry_recovery_candidates_prefers_baseline_over_scratch_prior(tmp_path):
    baseline = tmp_path / "geometry_baseline_dcm_155.res"
    scratch = tmp_path / "heat_dcm_155.0.overlap_a.res"
    segment = tmp_path / "heat_dcm_155.0.res"
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        prior_segment_restart=scratch,
        geometry_fallback_restarts=(segment,),
    )
    ladder = build_geometry_recovery_candidates(cfg)
    assert ladder == [baseline, segment]
    assert scratch not in ladder


def test_attempt_overlap_early_abort_recovery_uses_baseline_not_scratch(tmp_path):
    baseline = tmp_path / "geometry_baseline_dcm_155.res"
    baseline.write_text("baseline\n", encoding="utf-8")
    scratch = tmp_path / "heat_dcm_155.0.overlap_a.res"
    scratch.write_text("scratch\n", encoding="utf-8")
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        prior_segment_restart=scratch,
        geometry_fallback_restarts=(),
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.restore_geometry_from_ladder",
        return_value=baseline,
    ) as restore:
        ok = attempt_overlap_early_abort_recovery(
            cfg,
            chunk_nstep=250,
            steps_done=138,
            steps_before_chunk=0,
            overlap_context="heat segment 1/10",
        )
    assert ok is True
    restore.assert_called_once()
    called_candidates = restore.call_args[0][0]
    assert called_candidates[0] == baseline
    assert scratch not in called_candidates
