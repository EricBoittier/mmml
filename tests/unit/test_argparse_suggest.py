"""Fuzzy suggestions for unrecognized CLI option flags."""

from __future__ import annotations

import argparse

import pytest

from mmml.cli.argparse_suggest import (
    SuggestingArgumentParser,
    format_unrecognized_suggestions,
    option_strings_from_parser,
    suggest_close_option_flags,
)


def test_suggest_close_option_flags_finds_steps_per_recording():
    known = [
        "--jax-md-update-interval",
        "--jax-md-skin-distance",
        "--steps-per-recording",
        "--seed",
    ]
    pairs = suggest_close_option_flags(
        ["--steps-per-recordin", "800"],
        known,
        cutoff=0.5,
    )
    assert pairs
    assert pairs[0][0] == "--steps-per-recordin"
    assert "--steps-per-recording" in pairs[0][1]


def test_format_unrecognized_suggestions_appends_did_you_mean():
    hint = format_unrecognized_suggestions(
        "unrecognized arguments: --step-per-recording 800",
        ["--jax-md-update-interval", "--steps-per-recording", "--seed"],
    )
    assert hint is not None
    assert "Did you mean:" in hint
    assert "--steps-per-recording" in hint


def test_suggesting_argument_parser_error_includes_suggestion(capsys):
    p = SuggestingArgumentParser(prog="mmml md-system")
    p.add_argument("--steps-per-recording", type=int, default=100)
    p.add_argument("--jax-md-update-interval", type=int, default=1)
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--step-per-recording", "800"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err
    assert "Did you mean:" in err
    assert "--steps-per-recording" in err


def test_option_strings_from_parser_lists_long_flags():
    p = argparse.ArgumentParser()
    p.add_argument("--steps-per-recording", type=int)
    opts = option_strings_from_parser(p)
    assert "--steps-per-recording" in opts
