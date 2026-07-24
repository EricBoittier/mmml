"""Unit tests for pre-dynamics PyCHARMM lingo helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    apply_pre_dynamics_lingo_from_args,
    normalize_pycharmm_pre_dynamics_lingo,
    resolve_pre_dynamics_lingo_script,
    run_charmm_lingo_script,
    split_charmm_lingo_commands,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ([], ""),
        ("cons fix sele resid 1 end", "cons fix sele resid 1 end"),
        (
            ["cons fix sele resid 1 end", "umbr", "end"],
            "cons fix sele resid 1 end\numbr\nend",
        ),
        (["  ", "umbr", ""], "umbr"),
    ],
)
def test_normalize_pycharmm_pre_dynamics_lingo(value, expected: str) -> None:
    assert normalize_pycharmm_pre_dynamics_lingo(value) == expected


def test_normalize_rejects_bad_types() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        normalize_pycharmm_pre_dynamics_lingo(123)
    with pytest.raises(ValueError, match="list items must be strings"):
        normalize_pycharmm_pre_dynamics_lingo(["ok", 2])


def test_resolve_merges_inline_and_file(tmp_path: Path) -> None:
    path = tmp_path / "extra.inp"
    path.write_text("umbr\nend\n", encoding="utf-8")
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="cons fix sele resid 1 end",
        pycharmm_pre_dynamics_lingo_file=path,
    )
    assert resolve_pre_dynamics_lingo_script(args) == (
        "cons fix sele resid 1 end\numbr\nend"
    )


def test_split_charmm_lingo_commands_joins_continuations() -> None:
    script = """
    ! comment
    cons hmcm force 2.0 -
      sele all end
    open unit 44 write card name adumb-wuni.dat
    """
    assert split_charmm_lingo_commands(script) == [
        "cons hmcm force 2.0 sele all end",
        "open unit 44 write card name adumb-wuni.dat",
    ]


def test_apply_pre_dynamics_lingo_no_op_when_empty() -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=False,
        output_dir=None,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_not_called()


def test_apply_pre_dynamics_lingo_runs_via_helper(tmp_path: Path) -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="cons fix sele resid 1 end",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=True,
        output_dir=tmp_path,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_called_once()
    assert run.call_args.args[0] == "cons fix sele resid 1 end"
    assert run.call_args.kwargs["workdir"] == tmp_path


def test_apply_pre_dynamics_lingo_from_file(tmp_path: Path) -> None:
    path = tmp_path / "umbr.inp"
    path.write_text("umbr\nend\n", encoding="utf-8")
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=path,
        quiet=True,
        output_dir=tmp_path,
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.cli_common.run_charmm_lingo_script"
    ) as run:
        apply_pre_dynamics_lingo_from_args(args)
    run.assert_called_once()
    assert run.call_args.args[0] == "umbr\nend"
    assert run.call_args.kwargs["inp_path"] == path


def test_run_charmm_lingo_prefers_inp_file_api(tmp_path: Path) -> None:
    script = "cons fix sele resid 1 end\nopen unit 44 write card name adumb-wuni.dat"
    inp = tmp_path / "lingo.inp"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._eval_charmm_inp_file_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_inp_file",
        return_value=True,
    ) as invoke, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=inp, workdir=tmp_path)
    invoke.assert_called_once()
    script_fn.assert_not_called()
    assert inp.is_file()
    body = inp.read_text(encoding="utf-8")
    assert body.startswith("* MMML pre-dynamics")
    assert "cons fix sele resid 1 end" in body


def test_run_charmm_lingo_falls_back_to_line_commands(tmp_path: Path) -> None:
    script = "cons fix sele resid 1 end\nopen unit 44 write card name x.dat"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._eval_charmm_inp_file_available",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi_charmm_script",
        return_value=True,
    ) as script_fn:
        run_charmm_lingo_script(script, inp_path=tmp_path / "x.inp", workdir=tmp_path)
    assert script_fn.call_count == 2
    assert script_fn.call_args_list[0].args[0] == "cons fix sele resid 1 end"
    assert script_fn.call_args_list[1].args[0] == (
        "open unit 44 write card name x.dat"
    )