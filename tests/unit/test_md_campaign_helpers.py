"""Output-directory resolution and backend selection for MD campaigns.

``cli/run/md_campaign.py`` expands one YAML into many runs. Where each run's
results land is decided here, and getting it wrong is the expensive kind of
wrong: two runs writing to the same directory silently interleave trajectories
and overwrite each other's restarts, and a ``--output-dir`` that is accepted and
then ignored sends results somewhere the user is not looking.

The helpers covered here are pure dict/path logic -- no CHARMM, no MD.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from mmml.cli.run.md_campaign import (
    _campaign_needs_pycharmm,
    _explicit_cli_output_dir,
    _lookup_resolved_output_dir,
    _pycharmm_bonded_mm_mini_enabled,
    _resolve_output_dir,
    _unique_output_dir_if_exists,
)


# --- backend detection ------------------------------------------------------


def test_a_pycharmm_job_marks_the_campaign_as_needing_pycharmm():
    assert _campaign_needs_pycharmm({"runs": {"a": {"backend": "pycharmm"}}})


def test_a_jax_only_campaign_does_not_need_pycharmm():
    assert not _campaign_needs_pycharmm({"runs": {"a": {"backend": "jaxmd"}}})


def test_one_pycharmm_job_among_many_is_enough():
    campaign = {"runs": {"a": {"backend": "jaxmd"}, "b": {"backend": "pycharmm"}}}
    assert _campaign_needs_pycharmm(campaign)


def test_the_jobs_key_is_accepted_as_well_as_runs():
    assert _campaign_needs_pycharmm({"jobs": {"a": {"backend": "pycharmm"}}})


def test_an_empty_campaign_needs_nothing():
    assert not _campaign_needs_pycharmm({})


def test_a_null_job_body_does_not_crash():
    assert not _campaign_needs_pycharmm({"runs": {"a": None}})


# --- bonded-mm-mini opt-in --------------------------------------------------
#
# Documented as opt-in only: it can stall on PBC crystal free / CGENFF APPEND,
# so anything other than an explicit True must leave it off.


def test_bonded_mm_mini_is_off_by_default():
    assert not _pycharmm_bonded_mm_mini_enabled({}, {})


def test_bonded_mm_mini_can_be_enabled_per_job():
    assert _pycharmm_bonded_mm_mini_enabled({}, {"bonded_mm_mini": True})


def test_bonded_mm_mini_can_be_enabled_campaign_wide():
    assert _pycharmm_bonded_mm_mini_enabled({"bonded_mm_mini": True}, {})


def test_an_explicit_job_false_beats_a_campaign_true():
    """The disable check runs first, so the narrower scope wins."""
    assert not _pycharmm_bonded_mm_mini_enabled(
        {"bonded_mm_mini": True}, {"bonded_mm_mini": False}
    )


def test_an_explicit_campaign_false_beats_a_job_true():
    assert not _pycharmm_bonded_mm_mini_enabled(
        {"bonded_mm_mini": False}, {"bonded_mm_mini": True}
    )


@pytest.mark.parametrize("truthy", [1, "yes", "true"])
def test_only_a_real_boolean_enables_it(truthy):
    """A truthy string must not switch on a feature documented as opt-in."""
    assert not _pycharmm_bonded_mm_mini_enabled({}, {"bonded_mm_mini": truthy})


# --- output directory resolution -------------------------------------------


def test_an_explicit_output_dir_is_used_as_is(tmp_path):
    got = _resolve_output_dir({"output_dir": str(tmp_path / "here")}, "run-a")
    assert got == (tmp_path / "here").resolve()


def test_repeats_get_their_own_numbered_subdirectory(tmp_path):
    """Without this every repeat writes over the previous one."""
    merged = {"output_dir": str(tmp_path / "here"), "repeat": 3}
    assert _resolve_output_dir(merged, "run-a", rep=0).name == "rep00"
    assert _resolve_output_dir(merged, "run-a", rep=2).name == "rep02"


def test_a_single_repeat_does_not_add_a_subdirectory(tmp_path):
    merged = {"output_dir": str(tmp_path / "here"), "repeat": 1}
    assert _resolve_output_dir(merged, "run-a").name == "here"


def test_output_root_is_joined_with_the_run_id(tmp_path):
    got = _resolve_output_dir({"output_root": str(tmp_path / "root")}, "run-a")
    assert got == (tmp_path / "root" / "run-a").resolve()


def test_campaign_output_dir_is_used_when_no_root_is_set(tmp_path):
    got = _resolve_output_dir({"campaign_output_dir": str(tmp_path / "camp")}, "run-a")
    assert got == (tmp_path / "camp" / "run-a").resolve()


def test_the_default_root_is_results():
    assert _resolve_output_dir({}, "run-a") == (Path("results") / "run-a").resolve()


def test_output_dir_beats_output_root(tmp_path):
    merged = {
        "output_dir": str(tmp_path / "explicit"),
        "output_root": str(tmp_path / "root"),
    }
    assert _resolve_output_dir(merged, "run-a") == (tmp_path / "explicit").resolve()


# --- refusing to collapse several runs into one directory -------------------


def _args(**kw) -> Namespace:
    base = {"output_dir": None, "_cli_explicit": set()}
    base.update(kw)
    return Namespace(**base)


def test_no_explicit_flag_means_no_override():
    args = _args(output_dir="/tmp/x")  # present but not marked explicit
    assert _explicit_cli_output_dir(args, [("j", "j", 0)]) is None


def test_an_explicit_output_dir_for_a_single_run_is_honoured(tmp_path):
    args = _args(output_dir=str(tmp_path / "one"), _cli_explicit={"output_dir"})
    got = _explicit_cli_output_dir(args, [("j", "j", 0)])
    assert got == (tmp_path / "one").resolve()


def test_an_explicit_output_dir_for_several_runs_is_refused(tmp_path):
    """One directory cannot serve several runs; piling them up would interleave
    trajectories and clobber restarts."""
    args = _args(output_dir=str(tmp_path / "one"), _cli_explicit={"output_dir"})
    expanded = [("a", "run-a", 0), ("b", "run-b", 0)]

    with pytest.raises(ValueError, match="expands to 2 runs"):
        _explicit_cli_output_dir(args, expanded)


def test_the_refusal_names_the_runs_and_the_alternatives(tmp_path):
    args = _args(output_dir=str(tmp_path / "one"), _cli_explicit={"output_dir"})
    with pytest.raises(ValueError) as exc:
        _explicit_cli_output_dir(args, [("a", "run-a", 0), ("b", "run-b", 0)])
    msg = str(exc.value)
    assert "run-a" in msg and "run-b" in msg
    assert "--campaign-output-dir" in msg and "--job-id" in msg


def test_an_empty_output_dir_value_is_ignored():
    args = _args(output_dir="", _cli_explicit={"output_dir"})
    assert _explicit_cli_output_dir(args, [("j", "j", 0)]) is None


# --- not overwriting an existing run ----------------------------------------


def test_a_fresh_directory_is_returned_unchanged(tmp_path):
    target = tmp_path / "run"
    assert _unique_output_dir_if_exists(target, resume=False) == target.resolve()


def test_an_existing_directory_is_uniquified(tmp_path):
    """Reusing it would mix a new run's output into the old one's files."""
    target = tmp_path / "run"
    target.mkdir()

    got = _unique_output_dir_if_exists(target, resume=False)

    assert got != target.resolve()
    assert got.name.startswith("run_")
    assert not got.exists()


def test_resume_reuses_the_existing_directory(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    assert _unique_output_dir_if_exists(target, resume=True) == target.resolve()


def test_uniquification_keeps_the_parent(tmp_path):
    target = tmp_path / "nested" / "run"
    target.mkdir(parents=True)
    got = _unique_output_dir_if_exists(target, resume=False)
    assert got.parent == target.parent.resolve()


# --- looking a resolved path back up ----------------------------------------


def test_a_resolved_path_wins_over_the_static_layout(tmp_path):
    """After uniquification the in-run path is the truth, not the YAML."""
    resolved = {"run-a": tmp_path / "run-a_ab12"}
    got = _lookup_resolved_output_dir(resolved, {}, "run-a")
    assert got == tmp_path / "run-a_ab12"


def test_a_repeat_suffixed_run_id_is_matched_by_prefix(tmp_path):
    resolved = {"run-a.0": tmp_path / "run-a-rep0"}
    assert _lookup_resolved_output_dir(resolved, {}, "run-a") == tmp_path / "run-a-rep0"


def test_an_unknown_job_falls_back_to_the_static_layout():
    campaign = {"runs": {"run-b": {"output_root": "results"}}}
    got = _lookup_resolved_output_dir({}, campaign, "run-b")
    assert got.name == "run-b"


def test_an_exact_match_beats_a_prefix_match(tmp_path):
    resolved = {
        "run-a": tmp_path / "exact",
        "run-a.1": tmp_path / "prefixed",
    }
    assert _lookup_resolved_output_dir(resolved, {}, "run-a") == tmp_path / "exact"
