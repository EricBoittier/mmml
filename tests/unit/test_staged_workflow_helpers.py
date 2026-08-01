"""Restart/resume decision logic in the staged MD workflow.

``mlpot/staged_workflow.py`` is the most-edited file in the repo (230 commits in
90 days) and sat at 32.1%. It drives the heat -> NVE -> equi -> prod ladder, and
almost everything it decides is about *which restart file a stage resumes from*.

Those decisions fail quietly and expensively. Resume from the wrong ``.res`` and
the run silently discards hours of equilibration, or re-applies a cold-start
force gate to a finite-temperature liquid and FIREs it into a different
structure. Nothing raises; the trajectory is just wrong.

The functions covered here are the pure ones -- path selection, gate
predicates, resume rules -- which need neither CHARMM nor a GPU. The dynamics
calls they feed are exercised by the live ``pycharmm`` suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
    _can_seed_stage_from_memory,
    _equi_in_place_restart,
    _equi_restart_name,
    _heat_in_place_restart,
    _heat_restart_path,
    _is_dynamics_stage_restart_path,
    _prior_restart_for_stage,
    _restart_coord_read_candidates,
    _should_seed_heat_prior_restart,
    _should_skip_pre_dyn_fmax_gate,
    _trajectory_outputs,
    should_auto_resume_failed_staged_run,
)


def _paths(root: Path) -> dict[str, Path]:
    """The subset of ``staged_artifact_paths`` these helpers consult."""
    return {
        "heat_res": root / "heat.res",
        "nve_res": root / "nve.res",
        "equi_res": root / "equi.res",
        "prod_res": root / "prod.res",
        "geometry_baseline_res": root / "baseline.res",
    }


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class _IO:
    """Stand-in for CharmmTrajectoryFiles (only two fields are read)."""

    def __init__(self, read: Path | None, write: Path | None) -> None:
        self.restart_read = read
        self.restart_write = write


# --- segmented restart naming -----------------------------------------------


def test_unsegmented_equi_restart_is_the_plain_name():
    assert _equi_restart_name("tag", 1) == "equi.res"


def test_segmented_equi_restart_points_at_the_final_segment():
    """N segments write equi.0 ... equi.{N-1}; resuming must use the last."""
    assert _equi_restart_name("tag", 4) == "equi.3.res"


def test_heat_restart_is_the_plain_file_when_unsegmented(tmp_path):
    paths = _paths(tmp_path)
    assert _heat_restart_path(paths, "tag", 1) == paths["heat_res"]


def test_segmented_heat_restart_is_the_final_segment(tmp_path):
    paths = _paths(tmp_path)
    got = _heat_restart_path(paths, "tag", 3)
    assert got != paths["heat_res"]
    assert "2" in got.name, f"expected the last segment (index 2), got {got.name}"


# --- classifying a restart path ---------------------------------------------


@pytest.mark.parametrize("name", ["heat.res", "nve.res", "equi.res", "prod.res"])
def test_plain_stage_restarts_are_dynamics_restarts(tmp_path, name):
    assert _is_dynamics_stage_restart_path(tmp_path / name)


@pytest.mark.parametrize("name", ["equi.0.res", "prod.3.res", "nve.12.res"])
def test_segmented_stage_restarts_are_dynamics_restarts(tmp_path, name):
    assert _is_dynamics_stage_restart_path(tmp_path / name)


def test_none_is_not_a_dynamics_restart():
    assert not _is_dynamics_stage_restart_path(None)


@pytest.mark.parametrize("name", ["mini.res", "packmol.res", "something.res", "heat.dcd"])
def test_unrelated_paths_are_not_dynamics_restarts(tmp_path, name):
    assert not _is_dynamics_stage_restart_path(tmp_path / name)


def test_classification_is_case_insensitive(tmp_path):
    assert _is_dynamics_stage_restart_path(tmp_path / "EQUI.RES")


# --- the cold-start force gate ----------------------------------------------
#
# The |F|max ~2 eV/A ceiling is meant for Packmol/clash geometries. Applying it
# to a liquid that already finished heat would FIRE-minimise a valid
# finite-temperature structure into something else.


def test_gate_is_skipped_when_resuming_a_dynamics_restart(tmp_path):
    assert _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=False,
        dyn_stages=["equi"],
        restart_from=tmp_path / "nve.res",
    )


def test_gate_is_skipped_when_coords_were_seeded_from_a_restart():
    """Skip even when offline coord seeding failed -- EQUI CPT loads them anyway."""
    assert _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=True, dyn_stages=["equi"], restart_from=None
    )


def test_gate_is_skipped_for_a_memory_handoff_leg():
    assert _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=False,
        dyn_stages=["prod"],
        restart_from=None,
        handoff_coords_in_memory=True,
    )


def test_gate_still_applies_to_a_cold_heat_start(tmp_path):
    """heat is not a post-dynamics stage: the clash gate must stay armed."""
    assert not _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=True,
        dyn_stages=["heat"],
        restart_from=tmp_path / "equi.res",
    )


def test_gate_applies_when_there_is_nothing_to_resume():
    assert not _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=False, dyn_stages=["equi"], restart_from=None
    )


def test_gate_applies_with_no_stages_at_all():
    assert not _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=True, dyn_stages=[], restart_from=None
    )


@pytest.mark.parametrize("stage", ["equi", "nve", "prod"])
def test_every_post_dynamics_stage_can_skip_the_gate(tmp_path, stage):
    assert _should_skip_pre_dyn_fmax_gate(
        seeded_from_dynamics_restart=True, dyn_stages=[stage], restart_from=None
    )


# --- which restart a stage resumes from -------------------------------------


def test_heat_prefers_the_geometry_baseline(tmp_path):
    paths = _paths(tmp_path)
    _touch(paths["geometry_baseline_res"])
    got = _prior_restart_for_stage("heat", paths, restart_from=tmp_path / "other.res")
    assert got == paths["geometry_baseline_res"]


def test_heat_falls_back_to_an_explicit_restart(tmp_path):
    paths = _paths(tmp_path)
    explicit = tmp_path / "seed.res"
    assert _prior_restart_for_stage("heat", paths, restart_from=explicit) == explicit


def test_heat_has_no_prior_restart_when_nothing_is_available(tmp_path):
    assert _prior_restart_for_stage("heat", _paths(tmp_path), restart_from=None) is None


def test_nve_resumes_from_heat(tmp_path):
    paths = _paths(tmp_path)
    _touch(paths["heat_res"])
    assert _prior_restart_for_stage("nve", paths, restart_from=None) == paths["heat_res"]


def test_equi_prefers_nve_over_heat(tmp_path):
    """Both exist; resuming from heat would discard the whole NVE leg."""
    paths = _paths(tmp_path)
    _touch(paths["heat_res"])
    _touch(paths["nve_res"])
    assert _prior_restart_for_stage("equi", paths, restart_from=None) == paths["nve_res"]


def test_equi_falls_back_to_heat_when_nve_is_absent(tmp_path):
    paths = _paths(tmp_path)
    _touch(paths["heat_res"])
    assert _prior_restart_for_stage("equi", paths, restart_from=None) == paths["heat_res"]


def test_prod_resumes_from_equi(tmp_path):
    paths = _paths(tmp_path)
    _touch(paths["equi_res"])
    assert _prior_restart_for_stage("prod", paths, restart_from=None) == paths["equi_res"]


def test_prod_has_no_prior_restart_without_equi(tmp_path):
    assert _prior_restart_for_stage("prod", _paths(tmp_path), restart_from=None) is None


def test_an_explicit_restart_overrides_stage_defaults(tmp_path):
    paths = _paths(tmp_path)
    _touch(paths["nve_res"])
    explicit = tmp_path / "elsewhere.res"
    assert _prior_restart_for_stage("equi", paths, restart_from=explicit) == explicit


def test_a_missing_file_is_not_offered_as_a_prior_restart(tmp_path):
    """Paths are checked with is_file(); a stale name must not be returned."""
    assert _prior_restart_for_stage("nve", _paths(tmp_path), restart_from=None) is None


# --- in-place resume detection ----------------------------------------------


def test_heat_in_place_when_read_and_write_are_the_same_file(tmp_path):
    res = tmp_path / "heat.res"
    assert _heat_in_place_restart(_IO(res, res))


def test_heat_not_in_place_for_distinct_files(tmp_path):
    assert not _heat_in_place_restart(_IO(tmp_path / "a.res", tmp_path / "b.res"))


def test_in_place_detection_resolves_symlinks_and_dots(tmp_path):
    """``foo/../heat.res`` and ``heat.res`` are the same file."""
    res = _touch(tmp_path / "heat.res")
    indirect = tmp_path / "sub" / ".." / "heat.res"
    (tmp_path / "sub").mkdir(exist_ok=True)
    assert _heat_in_place_restart(_IO(indirect, res))


@pytest.mark.parametrize(("read", "write"), [(None, "x.res"), ("x.res", None), (None, None)])
def test_in_place_is_false_when_either_side_is_missing(tmp_path, read, write):
    io = _IO(tmp_path / read if read else None, tmp_path / write if write else None)
    assert not _heat_in_place_restart(io)
    assert not _equi_in_place_restart(io)


def test_equi_in_place_matches_heat_semantics(tmp_path):
    res = tmp_path / "equi.res"
    assert _equi_in_place_restart(_IO(res, res))


# --- trajectory outputs -----------------------------------------------------


def test_no_outputs_for_a_missing_trajectory(tmp_path):
    assert _trajectory_outputs(tmp_path / "absent.dcd") == []


def test_no_outputs_for_none():
    assert _trajectory_outputs(None) == []


def test_an_empty_dcd_is_not_an_output(tmp_path):
    """A zero-byte DCD means the stage produced nothing, not that it ran."""
    empty = tmp_path / "heat.dcd"
    empty.write_bytes(b"")
    assert _trajectory_outputs(empty) == []


def test_a_written_dcd_is_reported(tmp_path):
    dcd = tmp_path / "heat.dcd"
    dcd.write_bytes(b"CORD" + b"\x00" * 32)
    assert _trajectory_outputs(dcd) == [dcd]


# --- heat seeding / memory seeding ------------------------------------------


def test_first_heat_segment_from_live_state_needs_a_checkpoint():
    assert _should_seed_heat_prior_restart(
        seg_i=0, prev_restart_is_current_state=True, use_memory=False,
        memory_handoff_next=False,
    )


def test_memory_handoff_seeds_the_first_segment():
    assert _should_seed_heat_prior_restart(
        seg_i=0, prev_restart_is_current_state=False, use_memory=True,
        memory_handoff_next=False,
    )


def test_later_segments_do_not_reseed_without_a_handoff():
    assert not _should_seed_heat_prior_restart(
        seg_i=2, prev_restart_is_current_state=False, use_memory=False,
        memory_handoff_next=False,
    )


def test_memory_handoff_reseeds_a_later_segment():
    assert _should_seed_heat_prior_restart(
        seg_i=2, prev_restart_is_current_state=False, use_memory=True,
        memory_handoff_next=True,
    )


def test_memory_seeding_needs_an_invalid_on_disk_restart(tmp_path):
    """A truncated restart that matches live CHARMM state can be rewritten."""
    corrupt = _touch(tmp_path / "nve.res", "not a valid restart")
    assert _can_seed_stage_from_memory(
        corrupt, prev_restart=corrupt, prev_restart_is_current_state=True
    )


def test_memory_seeding_is_refused_when_state_is_stale(tmp_path):
    corrupt = _touch(tmp_path / "nve.res", "not a valid restart")
    assert not _can_seed_stage_from_memory(
        corrupt, prev_restart=corrupt, prev_restart_is_current_state=False
    )


def test_memory_seeding_is_refused_for_a_different_file(tmp_path):
    a = _touch(tmp_path / "nve.res", "bad")
    b = _touch(tmp_path / "heat.res", "bad")
    assert not _can_seed_stage_from_memory(
        a, prev_restart=b, prev_restart_is_current_state=True
    )


def test_memory_seeding_is_refused_when_nothing_is_on_disk(tmp_path):
    missing = tmp_path / "nve.res"
    assert not _can_seed_stage_from_memory(
        missing, prev_restart=missing, prev_restart_is_current_state=True
    )


# --- restart coordinate read candidates -------------------------------------


def test_read_candidates_always_include_the_requested_path(tmp_path):
    p = tmp_path / "equi.res"
    assert Path(p) in [Path(c) for c in _restart_coord_read_candidates(p)]


def test_read_candidates_are_deduplicated(tmp_path):
    got = _restart_coord_read_candidates(tmp_path / "equi.res")
    resolved = [str(Path(c)) for c in got]
    assert len(resolved) == len(set(resolved))


# --- auto-resume after a failed run -----------------------------------------


def _args(**kw) -> argparse.Namespace:
    base = dict(restart_from=None, rebuild_packmol=False, quiet=True)
    base.update(kw)
    return argparse.Namespace(**base)


def _summary(root: Path, exit_code: int) -> Path:
    return _touch(root / "stage_summary.json", json.dumps({"exit_code": exit_code}))


def test_a_failed_prior_run_triggers_auto_resume(tmp_path):
    _summary(tmp_path, 1)
    assert should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)


def test_a_successful_prior_run_does_not_resume(tmp_path):
    _summary(tmp_path, 0)
    assert not should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)


def test_no_summary_means_no_resume(tmp_path):
    assert not should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)


def test_an_explicit_restart_wins_over_auto_resume(tmp_path):
    _summary(tmp_path, 1)
    assert not should_auto_resume_failed_staged_run(
        _args(restart_from="seed.res"), out_dir=tmp_path
    )


def test_rebuild_packmol_refuses_to_inherit_a_stale_baseline(tmp_path):
    """Resuming here would discard the freshly packed geometry -- the documented
    reason this branch exists."""
    _summary(tmp_path, 1)
    assert not should_auto_resume_failed_staged_run(
        _args(rebuild_packmol=True), out_dir=tmp_path
    )


def test_a_corrupt_summary_does_not_resume(tmp_path):
    _touch(tmp_path / "stage_summary.json", "{not json")
    assert not should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)


def test_a_summary_without_an_exit_code_does_not_resume(tmp_path):
    _touch(tmp_path / "stage_summary.json", json.dumps({"stages": []}))
    assert not should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)


def test_a_non_numeric_exit_code_does_not_resume(tmp_path):
    _touch(tmp_path / "stage_summary.json", json.dumps({"exit_code": "boom"}))
    assert not should_auto_resume_failed_staged_run(_args(), out_dir=tmp_path)
