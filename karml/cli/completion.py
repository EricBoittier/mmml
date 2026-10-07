"""Shell tab-completion for the ``karml`` CLI (bash/zsh/fish via argcomplete)."""

from __future__ import annotations

import argparse
import os
import sys

from karml.cli.parser_utils import get_subcommand_parser
from karml.cli.registry import KARML_COMMANDS

_COMPLETION_SHELLS = ("bash", "zsh", "fish")


def try_autocomplete() -> bool:
    """If invoked by argcomplete, run completion and return True."""
    if not os.environ.get("_ARGCOMPLETE"):
        return False
    try:
        import argcomplete
    except ImportError:
        return False

    argv = sys.argv[1:]
    if argv and argv[0] == "completion":
        return False

    if argv and argv[0] not in ("commands", "examples", "completion"):
        sub = get_subcommand_parser(argv[0])
        if sub is not None:
            sys.argv = [sys.argv[0], *argv[1:]]
            argcomplete.autocomplete(sub)
            return True

    parser = argparse.ArgumentParser(prog="karml")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[c for c in KARML_COMMANDS if c != "completion"],
    )
    argcomplete.autocomplete(parser)
    return True


def _argcomplete_shellcode(executable: str, shell: str) -> str:
    import argcomplete

    if hasattr(argcomplete, "shellcode"):
        return argcomplete.shellcode([executable], shell=shell)
    from argcomplete.shell_integration import shellcode

    return shellcode([executable], shell=shell)


def _static_top_level_completion_script(shell: str) -> str:
    cmds = " ".join(c for c in KARML_COMMANDS if c != "completion")
    if shell == "bash":
        return f"""# KARML bash completion (top-level commands only; install argcomplete for full flags)
_karml_completions() {{
  local cur prev opts
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "{cmds}" -- "$cur") )
  fi
}}
complete -F _karml_completions karml
"""
    if shell == "zsh":
        zsh_cmds = " ".join(c for c in KARML_COMMANDS if c != "completion")
        return (
            "# KARML zsh completion (top-level commands only)\n"
            "compdef '_arguments \"*:command:((" + zsh_cmds.replace(" ", "\\n") + "))\"' karml\n"
        )
    if shell == "fish":
        fish_cmds = "\n".join(c for c in KARML_COMMANDS if c != "completion")
        return f"""# KARML fish completion (top-level commands only)
complete -c karml -f -n '__fish_use_subcommand' -a '{fish_cmds}'
"""
    raise ValueError(f"unsupported shell: {shell}")


def print_shell_completion(shell: str, *, executable: str = "karml") -> None:
    """Print a shell snippet that enables ``karml`` tab completion."""
    shell = shell.strip().lower()
    if shell not in _COMPLETION_SHELLS:
        raise SystemExit(
            f"Unknown shell {shell!r}; choose one of: {', '.join(_COMPLETION_SHELLS)}"
        )
    try:
        script = _argcomplete_shellcode(executable, shell)
    except ImportError:
        script = _static_top_level_completion_script(shell)
        print(
            "# Note: install argcomplete for full flag completion: pip install argcomplete",
            file=sys.stderr,
        )
    sys.stdout.write(script)
    if not script.endswith("\n"):
        sys.stdout.write("\n")


def completion_main(argv: list[str] | None = None) -> int:
    """``karml completion bash|zsh|fish [--executable NAME]``."""
    parser = argparse.ArgumentParser(
        prog="karml completion",
        description="Print shell completion script for karml (pipe into source or eval).",
    )
    parser.add_argument(
        "shell",
        choices=_COMPLETION_SHELLS,
        help="Target shell",
    )
    parser.add_argument(
        "--executable",
        default="karml",
        help="CLI executable name (default: karml)",
    )
    parser.add_argument(
        "--install-hint",
        action="store_true",
        help="Print setup instructions after the script",
    )
    args = parser.parse_args(argv)
    print_shell_completion(args.shell, executable=args.executable)
    if args.install_hint:
        shell = args.shell
        print(
            f"\n# Setup ({shell}):\n"
            f"#   eval \"$({args.executable} completion {shell})\"\n"
            f"# or, with argcomplete installed:\n"
            f"#   eval \"$(register-python-argcomplete {args.executable})\"\n",
            file=sys.stderr,
        )
    return 0
