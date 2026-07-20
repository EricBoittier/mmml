from __future__ import annotations

import argparse
import io

from mmml.cli.help_style import (
    group_parser_options,
    install_colored_argparse,
    print_cli_text,
)


def _flat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmml demo")
    parser.add_argument("residue")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--calculator")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--backend")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def test_flat_parser_is_grouped_by_function():
    parser = _flat_parser()
    group_parser_options(parser)
    text = parser.format_help()
    for heading in (
        "Input & configuration:",
        "Scientific model:",
        "Execution:",
        "Output & artifacts:",
        "Diagnostics & safety:",
    ):
        assert heading in text
    assert text.rindex("--checkpoint") < text.index("Scientific model:")
    assert text.rindex("--temperature") > text.index("Scientific model:")


def test_explicit_argument_groups_are_preserved():
    parser = argparse.ArgumentParser(prog="mmml explicit")
    custom = parser.add_argument_group("Physics controls")
    custom.add_argument("--cutoff")
    for index in range(5):
        parser.add_argument(f"--option-{index}")
    group_parser_options(parser)
    text = parser.format_help()
    assert "Physics controls:" in text
    assert "Input & configuration:" not in text


def test_cli_text_is_colored_only_when_enabled(monkeypatch):
    plain = io.StringIO()
    monkeypatch.setenv("MMML_NO_RICH", "1")
    print_cli_text("usage: mmml demo --config CONFIG\n", stream=plain)
    assert "\x1b[" not in plain.getvalue()

    colored = io.StringIO()
    monkeypatch.delenv("MMML_NO_RICH", raising=False)
    monkeypatch.setenv("MMML_RICH", "1")
    print_cli_text("usage: mmml demo --config CONFIG\n", stream=colored)
    assert "\x1b[" in colored.getvalue()
    assert "--config" in colored.getvalue()

    error = io.StringIO()
    print_cli_text("mmml demo: error: bad option\n", stream=error)
    assert "\x1b[" in error.getvalue()


def test_argparse_install_groups_help_without_changing_plain_content(monkeypatch):
    monkeypatch.setenv("MMML_NO_RICH", "1")
    install_colored_argparse()
    text = _flat_parser().format_help()
    assert "Input & configuration:" in text
    assert "--output" in text
