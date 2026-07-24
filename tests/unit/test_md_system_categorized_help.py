"""Categorized ``mmml md-system -h`` / ``-hN`` / ``--help-all``."""

from __future__ import annotations

import contextlib
import io

import pytest

from mmml.cli.md_system_help import (
    MD_SYSTEM_HELP_CATEGORIES,
    _CORE_DESTS,
    category_titles,
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


def test_parse_help_mode_aliases():
    assert parse_help_mode(["-hcore"]) == 1
    assert parse_help_mode(["-hbuilders"]) == 2
    assert parse_help_mode(["-hbox"]) == 2
    assert parse_help_mode(["--help-pycharmm"]) == 4
    assert parse_help_mode(["--help=overlap"]) == 5
    assert parse_help_mode(["-hmin"]) == 6
    assert parse_help_mode(["-hhybrid"]) == 7
    assert parse_help_mode(["--help-lambda"]) == 8
    assert parse_help_mode(["-hnope"]) == "?nope"


def test_help_index_is_short():
    parser = md_system.build_parser()
    text = format_help_index(parser)
    assert "-h1" in text
    assert "core" in text
    assert "pycharmm" in text
    assert "Core setup" in text
    assert "--help-all" in text
    assert text.count("--dynamics-overlap-action") == 0
    # Index must stay a short menu, not the old description wall.
    assert len(text.splitlines()) < 30
    assert "lambda TI for arbitrary" not in text


def test_help_category_one_is_core_only():
    parser = md_system.build_parser()
    text = format_help_category(parser, 1)
    assert "1. Core setup" in text
    assert "--setup" in text
    assert "--backend" in text
    assert "--composition" in text
    # Must not swallow other categories (the double-underscore prefix bug).
    assert "--heat-thermostat" not in text
    assert "--dynamics-overlap-action" not in text
    assert "--evaluate-npz" not in text
    assert "--flat-bottom-radius" not in text
    buckets = iter_categorized_actions(parser)
    assert len(buckets[1]) <= len(_CORE_DESTS)
    assert len(buckets[1]) <= 30


def test_help_category_five_has_overlap_guard():
    parser = md_system.build_parser()
    text = format_help_category(parser, 5)
    assert "--dynamics-overlap-action" in text
    assert "--setup" not in text


def test_prefix_trailing_underscore_matches():
    """``heat_`` must match ``heat_ihtfrq`` (not ``heat__ihtfrq``)."""
    parser = md_system.build_parser()
    by_dest = {a.dest: a for a in parser._actions if a.option_strings}
    assert classify_action(parser, by_dest["heat_ihtfrq"]) == 4
    assert classify_action(parser, by_dest["flat_bottom_radius"]) == 3
    assert classify_action(parser, by_dest["dcd_nsavc"]) == 4
    assert classify_action(parser, by_dest["bonded_mm_mini"]) == 4
    assert classify_action(parser, by_dest["ml_batch_size"]) == 7
    assert classify_action(parser, by_dest["evaluate_npz"]) == 8


def test_longest_prefix_wins_for_nested_dests():
    parser = md_system.build_parser()
    by_dest = {a.dest: a for a in parser._actions if a.option_strings}
    assert classify_action(parser, by_dest["quiet"]) == 4
    assert classify_action(parser, by_dest["quiet_bfgs"]) == 6
    assert classify_action(parser, by_dest["verbose"]) == 4
    assert classify_action(parser, by_dest["verbose_bfgs"]) == 6
    assert classify_action(parser, by_dest["ps"]) == 1
    assert classify_action(parser, by_dest["ps_heat"]) == 4


def test_help_all_contains_every_category_heading():
    parser = md_system.build_parser()
    text = format_help_all(parser)
    for num, title, aliases in MD_SYSTEM_HELP_CATEGORIES:
        assert f"{num}. {title}" in text
        for alias in aliases:
            assert f"-h{alias}" in text
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
        assert cat in category_titles()


def test_build_parser_h_prints_index(capsys):
    with pytest.raises(SystemExit) as excinfo:
        md_system.build_parser().parse_args(["-h"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Help is split by category" in out
    assert "-h1" in out
    assert out.count("--dynamics-overlap-action") == 0


def test_build_parser_h4_prints_pycharmm_category(capsys):
    with pytest.raises(SystemExit) as excinfo:
        md_system.build_parser().parse_args(["-h4"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "4. PyCHARMM" in out
    assert "--md-stages" in out or "--ps-heat" in out or "--echeck" in out


def test_build_parser_halias_matches_number(capsys):
    with pytest.raises(SystemExit) as excinfo:
        md_system.build_parser().parse_args(["-hpycharmm"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "4. PyCHARMM" in out
    assert "-hpycharmm" in out


def test_main_help_short_circuits_to_index(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["mmml md-system", "-h"])
    with pytest.raises(SystemExit) as excinfo:
        md_system.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Help is split by category" in out


def test_unknown_help_category_errors():
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        with pytest.raises(SystemExit) as excinfo:
            md_system.build_parser().parse_args(["-h99"])
    assert excinfo.value.code == 2
    err = buf.getvalue()
    assert "unknown help category" in err
