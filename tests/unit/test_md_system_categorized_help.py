"""Categorized ``mmml md-system -h`` / ``-hN`` / ``--help-all``."""

from __future__ import annotations

import contextlib
import io

import pytest

from mmml.cli.md_system_help import (
    MD_SYSTEM_HELP_CATEGORIES,
    classify_action,
    format_help_all,
    format_help_category,
    format_help_index,
    iter_categorized_actions,
    parse_help_mode,
)
from mmml.cli.run import md_system


def test_parse_help_mode_tokens():
    assert parse_help_mode(["-h"]) == "index"
    assert parse_help_mode(["--help"]) == "index"
    assert parse_help_mode(["-h1"]) == 1
    assert parse_help_mode(["--help-4"]) == 4
    assert parse_help_mode(["--help=8"]) == 8
    assert parse_help_mode(["--help-all"]) == "all"
    assert parse_help_mode(["-ha"]) == "all"
    assert parse_help_mode(["--setup", "pbc_nve"]) is None


def test_help_index_lists_categories_not_flag_wall():
    parser = md_system.build_parser()
    text = format_help_index(parser)
    assert "-h1" in text
    assert "Core setup" in text
    assert "--help-all" in text
    # Index must stay short: no full option dump.
    assert text.count("--dynamics-overlap-action") == 0
    assert "--setup" in text  # listed under common starting flags


def test_help_category_one_has_core_flags():
    parser = md_system.build_parser()
    text = format_help_category(parser, 1)
    assert "1. Core setup" in text
    assert "--setup" in text
    assert "--backend" in text
    assert "--composition" in text


def test_help_category_five_has_overlap_guard():
    parser = md_system.build_parser()
    text = format_help_category(parser, 5)
    assert "--dynamics-overlap-action" in text
    assert "--setup" not in text


def test_help_all_contains_every_category_heading():
    parser = md_system.build_parser()
    text = format_help_all(parser)
    for num, title in MD_SYSTEM_HELP_CATEGORIES:
        assert f"{num}. {title}" in text
    assert "--setup" in text
    assert "--dynamics-overlap-action" in text
    assert "--box-size" in text


def test_every_option_is_classified():
    parser = md_system.build_parser()
    buckets = iter_categorized_actions(parser)
    classified = {id(a) for actions in buckets.values() for a in actions}
    for action in parser._actions:
        if not action.option_strings:
            continue
        if getattr(action, "help", None) is __import__("argparse").SUPPRESS:
            continue
        assert id(action) in classified, f"unclassified: {action.option_strings}"
        cat = classify_action(parser, action)
        assert cat in dict(MD_SYSTEM_HELP_CATEGORIES)


def test_build_parser_h_prints_index(capsys):
    with pytest.raises(SystemExit) as excinfo:
        md_system.build_parser().parse_args(["-h"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Help is split into categories" in out
    assert "-h1" in out
    assert out.count("--dynamics-overlap-action") == 0


def test_build_parser_h4_prints_pycharmm_category(capsys):
    with pytest.raises(SystemExit) as excinfo:
        md_system.build_parser().parse_args(["-h4"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "4. PyCHARMM" in out
    assert "--md-stages" in out or "--ps-heat" in out or "--echeck" in out


def test_main_help_short_circuits_to_index(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["mmml md-system", "-h"])
    with pytest.raises(SystemExit) as excinfo:
        md_system.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Help is split into categories" in out


def test_unknown_help_category_errors():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with pytest.raises(SystemExit) as excinfo:
            md_system.build_parser().parse_args(["-h99"])
    assert excinfo.value.code == 2
    err = buf.getvalue()
    assert "unknown help category" in err
