"""Tests for geometry checkpoint ladder and pretreat resume."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from tests.unit.conftest import write_minimal_restart

from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
    attempt_overlap_early_abort_recovery,
    build_extent_recovery_candidates,
    build_geometry_recovery_candidates,
    discover_resume_restart,
    first_valid_geometry_crd_path,
    first_valid_restart_path,
    is_geometry_recovery_crd_path,
    is_handoff_seed_restart_path,
    is_heat_segment_restart_path,
    is_overlap_scratch_restart_path,
    pretreat_stage_complete,
    resolve_geometry_checkpoint_ladder,
    restore_geometry_from_ladder,
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
    write_minimal_restart(paths["geometry_baseline_res"])
    write_minimal_restart(paths["charmm_mm_prod_res"])

    ladder = resolve_geometry_checkpoint_ladder(paths, tag, n_heat_segments=1)
    assert ladder.index(paths["geometry_baseline_res"]) < ladder.index(
        paths["charmm_mm_prod_res"]
    )
    found = first_valid_restart_path(ladder)
    assert found == paths["geometry_baseline_res"].resolve()


def test_discover_resume_restart_prefers_heat_segment(tmp_path):
    tag = "dcm_10"
    heat_seg = tmp_path / "heat.1.res"
    write_minimal_restart(heat_seg)
    baseline = tmp_path / f"geometry_baseline_{tag}.res"
    write_minimal_restart(baseline)
    paths = {
        "heat_res": tmp_path / f"heat_{tag}.res",
        "geometry_baseline_res": baseline,
    }
    found = discover_resume_restart(tmp_path, tag, paths=paths, n_heat_segments=2)
    assert found == heat_seg.resolve()


def test_discover_resume_restart_prefers_baseline_over_pretreat(tmp_path):
    tag = "dcm_52"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    prod = pretreat / f"charmm_mm_prod_{tag}.res"
    write_minimal_restart(prod)
    baseline = tmp_path / f"geometry_baseline_{tag}.res"
    write_minimal_restart(baseline)
    paths = {
        "heat_res": tmp_path / f"heat_{tag}.res",
        "charmm_mm_prod_res": prod,
        "geometry_baseline_res": baseline,
    }
    found = discover_resume_restart(tmp_path, tag, paths=paths, n_heat_segments=10)
    assert found == baseline.resolve()


def test_handoff_seed_excluded_from_recovery_ladder(tmp_path):
    handoff = tmp_path / "handoff" / "continue_seed.res"
    handoff.parent.mkdir()
    handoff.write_text("seed\n", encoding="utf-8")
    baseline = tmp_path / "geometry_baseline_dcm.res"
    write_minimal_restart(baseline)
    overlap = DynamicsOverlapConfig(
        geometry_baseline_restart=baseline,
        geometry_fallback_restarts=(handoff,),
    )
    cands = build_geometry_recovery_candidates(overlap)
    assert handoff.resolve() not in [c.resolve() for c in cands]
    assert is_handoff_seed_restart_path(handoff)


def test_pretreat_resume_skips_completed_heat(tmp_path):
    tag = "dcm_190"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    heat = pretreat / f"charmm_mm_heat_{tag}.res"
    write_minimal_restart(heat)
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
    write_minimal_restart(prod)
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
    expected = out / "baseline.res"
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
            return_value=True,
        ) as rewrite,
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._valid_restart_file",
            return_value=expected,
        ),
    ):
        path = write_geometry_baseline_restart(out, "dcm_10")
    rewrite.assert_called_once()
    assert path == expected.resolve()


def test_write_geometry_baseline_restart_unlinks_invalid_file(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.rewrite_dynamics_restart_validated",
            return_value=True,
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics._valid_restart_file",
            return_value=None,
        ),
    ):
        path = write_geometry_baseline_restart(out, "dcm_10")
    assert path is None
    assert not (out / "baseline.res").exists()


def test_pretreat_resume_continues_partial_heat(tmp_path):
    tag = "aco_266"
    pretreat = tmp_path / "pretreat"
    pretreat.mkdir()
    heat = pretreat / f"charmm_mm_heat_{tag}.res"
    write_minimal_restart(heat)
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
        charmm_mm_pretreat_dt_fs=0.2,
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
    assert is_overlap_scratch_restart_path("heat.0.overlap_a.res")
    assert is_overlap_scratch_restart_path("/tmp/heat.a.res")
    assert is_overlap_scratch_restart_path("/tmp/heat.overlap_b.res")
    assert not is_overlap_scratch_restart_path("heat.0.res")
    assert not is_overlap_scratch_restart_path("baseline.res")


def test_is_heat_segment_restart_path():
    assert is_heat_segment_restart_path("heat.0.res")
    assert is_heat_segment_restart_path("heat.12.res")
    assert is_heat_segment_restart_path("heat_dcm_155.0.res")
    assert not is_heat_segment_restart_path("heat.res")
    assert not is_heat_segment_restart_path("heat.a.res")
    assert not is_heat_segment_restart_path("baseline.res")
    assert not is_heat_segment_restart_path("equi.res")


def test_build_extent_recovery_candidates_skips_heat_segment_tails(tmp_path):
    baseline = tmp_path / "baseline.res"
    crd = tmp_path / "03_bonded_mm_after_mini_dcm.crd"
    heat_seg = tmp_path / "heat.0.res"
    equi = tmp_path / "equi.res"
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        prior_segment_restart=heat_seg,
        geometry_fallback_restarts=(crd, heat_seg, equi),
    )
    ladder = build_extent_recovery_candidates(cfg)
    assert ladder == [baseline, crd, equi]
    assert heat_seg not in ladder


def test_restore_geometry_from_ladder_extent_prefers_crd_over_heat_segment(
    tmp_path,
):
    baseline = tmp_path / "baseline.res"
    baseline.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    crd = tmp_path / "03_bonded_mm_after_mini.crd"
    crd.write_text("crd coords\n", encoding="utf-8")
    heat = tmp_path / "heat.0.res"
    write_minimal_restart(heat)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        prior_segment_restart=heat,
        geometry_fallback_restarts=(crd, heat),
    )
    candidates = build_extent_recovery_candidates(cfg)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_crd"
    ) as restore_crd:
        path = restore_geometry_from_ladder(
            candidates,
            label="Fly-off recovery",
        )
    assert path == crd.resolve()
    restore_crd.assert_called_once_with(crd.resolve())


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


def test_build_geometry_recovery_candidates_skips_pretreat_mm(tmp_path):
    baseline = tmp_path / "geometry_baseline_dcm_254.res"
    pretreat = tmp_path / "pretreat" / "charmm_mm_prod_dcm_254.res"
    pretreat.parent.mkdir()
    heat = tmp_path / "heat_dcm_254.0.res"
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        geometry_fallback_restarts=(pretreat, heat),
    )
    ladder = build_geometry_recovery_candidates(cfg)
    assert ladder == [baseline, heat]
    assert pretreat not in ladder


def test_early_abort_trust_in_memory_rejects_cpt_cold_start_blowup():
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        _early_abort_trust_in_memory,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
    )

    cfg = DynamicsOverlapConfig(action="rescue", n_monomers=13)
    ctx = mock.Mock()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_memory_coordinates_usable",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=5692.0,
    ):
        assert not _early_abort_trust_in_memory(
            cfg,
            integrated=1,
            chunk_nstep=640,
            chunk_index=0,
            cpt=True,
            mlpot_ctx=ctx,
        )


def test_early_abort_trust_in_memory_accepts_stable_heat_abort():
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        _early_abort_trust_in_memory,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        DynamicsOverlapConfig,
    )

    cfg = DynamicsOverlapConfig(action="rescue", n_monomers=13)
    ctx = mock.Mock()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_memory_coordinates_usable",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.resolve_mlpot_grms_kcalmol_A",
        return_value=1.7,
    ):
        assert _early_abort_trust_in_memory(
            cfg,
            integrated=383,
            chunk_nstep=500,
            chunk_index=5,
            cpt=True,
            mlpot_ctx=ctx,
        )


def test_attempt_overlap_early_abort_recovery_reports_memory_source(tmp_path):
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_fallback_restarts=(),
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint.restore_geometry_from_ladder",
        side_effect=RuntimeError("no disk"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint._early_abort_trust_in_memory",
        return_value=True,
    ):
        recovery = attempt_overlap_early_abort_recovery(
            cfg,
            chunk_nstep=640,
            steps_done=2706,
            steps_before_chunk=2560,
            overlap_context="PROD",
            mlpot_ctx=mock.Mock(),
        )
    assert recovery.ok is True
    assert recovery.source == "memory"


def test_build_early_abort_recovery_candidates_prefers_overlap_read(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        build_early_abort_recovery_candidates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig

    baseline = tmp_path / "geometry_baseline_dcm_10.res"
    write_minimal_restart(baseline)
    heat = tmp_path / "heat_dcm_10.res"
    write_minimal_restart(heat)
    scratch = tmp_path / "equi_dcm_10.overlap_a.res"
    write_minimal_restart(scratch)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        prior_segment_restart=heat,
        geometry_fallback_restarts=(),
    )
    ladder = build_early_abort_recovery_candidates(
        cfg,
        overlap_restart_read=scratch,
    )
    assert ladder[0] == scratch.resolve()
    assert heat in ladder
    assert baseline in ladder


def test_build_early_abort_recovery_candidates_includes_segment_restart(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        build_early_abort_recovery_candidates,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import DynamicsOverlapConfig

    equi = tmp_path / "equi_dcm_10.res"
    write_minimal_restart(equi)
    baseline = tmp_path / "geometry_baseline_dcm_10.res"
    write_minimal_restart(baseline)
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        geometry_fallback_restarts=(),
    )
    ladder = build_early_abort_recovery_candidates(
        cfg,
        segment_restart_read=equi,
    )
    assert ladder[0] == equi.resolve()
    assert baseline in ladder


def test_attempt_overlap_early_abort_recovery_uses_baseline_without_overlap_read(tmp_path):
    """Without overlap_restart_read, scratch prior_segment is still excluded."""
    baseline = tmp_path / "geometry_baseline_dcm_155.res"
    write_minimal_restart(baseline)
    scratch = tmp_path / "heat_dcm_155.0.overlap_a.res"
    write_minimal_restart(scratch)
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
        recovery = attempt_overlap_early_abort_recovery(
            cfg,
            chunk_nstep=250,
            steps_done=138,
            steps_before_chunk=0,
            overlap_context="heat segment 1/10",
        )
    assert recovery.ok is True
    assert recovery.source == "restart"
    restore.assert_called_once()
    called_candidates = restore.call_args[0][0]
    assert called_candidates[0] == baseline
    assert scratch not in called_candidates


def test_first_valid_geometry_crd_path_skips_restart_files(tmp_path):
    junk = (
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n"
    )
    baseline = tmp_path / "geometry_baseline_dcm_25.res"
    baseline.write_text(junk, encoding="utf-8")
    crd = tmp_path / "03_bonded_mm_after_mini_dcm_25.crd"
    crd.write_text("crd coords\n", encoding="utf-8")
    heat = tmp_path / "heat_dcm_25.res"
    heat.write_text(junk, encoding="utf-8")
    ladder = [baseline, crd, heat]

    assert first_valid_restart_path(ladder) is None
    assert first_valid_geometry_crd_path(ladder) == crd.resolve()
    assert is_geometry_recovery_crd_path(crd)


def test_restore_geometry_from_ladder_falls_back_to_crd(tmp_path):
    crd = tmp_path / "03_bonded_mm_after_mini_dcm_25.crd"
    crd.write_text("crd coords\n", encoding="utf-8")
    bad = tmp_path / "heat_dcm_25.res"
    bad.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    candidates = [bad, crd]

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_crd"
    ) as restore_crd:
        path = restore_geometry_from_ladder(candidates, label="test recovery")

    restore_crd.assert_called_once_with(crd.resolve())
    assert path == crd.resolve()


def test_restore_geometry_from_ladder_falls_back_to_in_memory(tmp_path):
    bad = tmp_path / "heat_dcm_25.res"
    bad.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    candidates = [bad]

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_memory_coordinates_usable",
        return_value=True,
    ):
        path = restore_geometry_from_ladder(
            candidates,
            label="test recovery",
            allow_in_memory=True,
        )

    assert path == Path("<in-memory>")


def test_attempt_overlap_early_abort_recovery_uses_crd_when_restarts_invalid(
    tmp_path,
):
    baseline = tmp_path / "geometry_baseline_dcm_25.res"
    baseline.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    crd = tmp_path / "03_bonded_mm_after_mini_dcm_25.crd"
    crd.write_text("crd coords\n", encoding="utf-8")
    heat = tmp_path / "heat_dcm_25.res"
    heat.write_text(
        "NOTE!! THIS FILE  C A N N O T  BE USED TO RESTART A RUN!!!\n",
        encoding="utf-8",
    )
    cfg = DynamicsOverlapConfig(
        action="rescue",
        n_monomers=2,
        geometry_baseline_restart=baseline,
        geometry_fallback_restarts=(crd, heat),
    )

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.restore_charmm_state_from_crd"
    ) as restore_crd, mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery.charmm_memory_coordinates_usable",
        return_value=False,
    ):
        recovery = attempt_overlap_early_abort_recovery(
            cfg,
            chunk_nstep=400,
            steps_done=1,
            steps_before_chunk=0,
            overlap_context="HEAT",
        )

    assert recovery.ok is True
    assert recovery.source == "crd"
    restore_crd.assert_called_once_with(crd.resolve())