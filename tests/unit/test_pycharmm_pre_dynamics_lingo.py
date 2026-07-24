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


def test_apply_pre_dynamics_lingo_no_op_when_empty() -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=False,
    )
    fake_pycharmm = mock.Mock()
    with mock.patch.dict("sys.modules", {"pycharmm": fake_pycharmm}):
        apply_pre_dynamics_lingo_from_args(args)
    fake_pycharmm.lingo.charmm_script.assert_not_called()


def test_apply_pre_dynamics_lingo_runs_script() -> None:
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="cons fix sele resid 1 end",
        pycharmm_pre_dynamics_lingo_file=None,
        quiet=True,
    )
    fake_pycharmm = mock.Mock()
    with mock.patch.dict("sys.modules", {"pycharmm": fake_pycharmm}):
        apply_pre_dynamics_lingo_from_args(args)
    fake_pycharmm.lingo.charmm_script.assert_called_once_with(
        "cons fix sele resid 1 end"
    )


def test_apply_pre_dynamics_lingo_from_file(tmp_path: Path) -> None:
    path = tmp_path / "umbr.inp"
    path.write_text("umbr\nend\n", encoding="utf-8")
    args = argparse.Namespace(
        pycharmm_pre_dynamics_lingo="",
        pycharmm_pre_dynamics_lingo_file=path,
        quiet=True,
    )
    fake_pycharmm = mock.Mock()
    with mock.patch.dict("sys.modules", {"pycharmm": fake_pycharmm}):
        apply_pre_dynamics_lingo_from_args(args)
    fake_pycharmm.lingo.charmm_script.assert_called_once_with("umbr\nend")
