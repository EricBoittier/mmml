"""Tests for mmml shell tab-completion helpers."""

from __future__ import annotations

import argparse

from mmml.cli import __main__ as cli_main
from mmml.cli.completion import completion_main, get_subcommand_parser, print_shell_completion
from mmml.cli.parser_utils import get_subcommand_parser as parser_for
from mmml.cli.registry import _DISPATCH_COMMANDS


def test_mmml_commands_match_dispatch():
    dispatch = set(cli_main._DISPATCH_COMMANDS)
    assert dispatch == set(_DISPATCH_COMMANDS)


def test_md_system_subcommand_parser_builds():
    parser = parser_for("md-system")
    assert parser is not None
    assert any(
        a.dest == "setup"
        for a in parser._actions
        if isinstance(a, argparse._StoreAction)
    )


def test_completion_main_bash_prints_script(capsys):
    assert completion_main(["bash"]) == 0
    out = capsys.readouterr().out
    assert "mmml" in out
    assert "md-system" in out or "complete" in out


def test_static_bash_completion_when_no_argcomplete(monkeypatch):
    import mmml.cli.completion as completion_mod

    def _raise_import(*_a, **_k):
        raise ImportError("argcomplete not installed")

    monkeypatch.setattr(completion_mod, "_argcomplete_shellcode", _raise_import)
    print_shell_completion("bash")
    # no exception


def test_unknown_shell_exits():
    try:
        completion_main(["powershell"])
        raised = False
    except SystemExit as exc:
        raised = exc.code != 0
    assert raised
