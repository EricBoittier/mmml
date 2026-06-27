"""Shell tab-completion for the ``mmml`` CLI (bash/zsh/fish via argcomplete)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from typing import Any

# Keep in sync with ``mmml.cli.__main__`` dispatch table.
MMML_COMMANDS: tuple[str, ...] = (
    "make-res",
    "make-box",
    "build-crystal",
    "run",
    "md-system",
    "liquid-box",
    "lambda-mbar",
    "run-pycharmm",
    "pycharmm-two-residue-sample",
    "xml2npz",
    "validate",
    "train",
    "train-joint",
    "evaluate",
    "downstream",
    "fix-and-split",
    "pyscf-dft",
    "pyscf-mp2",
    "pyscf-evaluate",
    "pyscf-evaluate-mp2",
    "verify-esp-alignment",
    "normal-mode-sample",
    "physnet-train",
    "physnet-md",
    "physnet-evaluate",
    "compare-npz",
    "cross-check",
    "ef-train",
    "ef-evaluate",
    "ef-md",
    "active-learning",
    "kernel-fit",
    "interpolate-xyz",
    "unwrap-traj",
    "sample-diverse-xyz",
    "gui",
    "extract-checkpoint-metrics",
    "orbax-to-json",
    "orca-server",
    "orca-client",
    "orca-external",
    "completion",
)

_COMPLETION_SHELLS = ("bash", "zsh", "fish")


def build_top_level_parser() -> argparse.ArgumentParser:
    """Parser for ``mmml <command>`` (used for top-level command completion)."""
    return argparse.ArgumentParser(prog="mmml")
    # argcomplete uses add_argument for choices; attach below in try_autocomplete


def _md_system_parser() -> argparse.ArgumentParser:
    from mmml.cli.run.md_system import build_parser

    return build_parser()


def _physnet_train_parser() -> argparse.ArgumentParser:
    from mmml.cli.make.make_training import build_parser

    return build_parser()


def _lambda_mbar_parser() -> argparse.ArgumentParser:
    from mmml.cli.run.lambda_mbar import build_parser

    return build_parser()


def _run_parser() -> argparse.ArgumentParser:
    from mmml.cli.run.run_sim import build_parser

    return build_parser()


def _make_res_parser() -> argparse.ArgumentParser:
    from mmml.cli.make.make_res import build_parser

    return build_parser()


def _run_pycharmm_parser() -> argparse.ArgumentParser:
    from mmml.cli.run.run_pycharmm import build_parser

    return build_parser()


def _active_learning_parser() -> argparse.ArgumentParser:
    from mmml.cli.misc.active_learning import build_parser

    return build_parser()


def _cross_check_parser() -> argparse.ArgumentParser:
    from mmml.cli.misc.cross_check import build_parser

    return build_parser()


_SUBCOMMAND_PARSER_BUILDERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "md-system": _md_system_parser,
    "physnet-train": _physnet_train_parser,
    "lambda-mbar": _lambda_mbar_parser,
    "run": _run_parser,
    "make-res": _make_res_parser,
    "run-pycharmm": _run_pycharmm_parser,
    "active-learning": _active_learning_parser,
    "cross-check": _cross_check_parser,
}


def get_subcommand_parser(command: str) -> argparse.ArgumentParser | None:
    """Return an ``ArgumentParser`` for ``mmml <command>`` when wired for completion."""
    builder = _SUBCOMMAND_PARSER_BUILDERS.get(command)
    if builder is None:
        return None
    try:
        return builder()
    except Exception:
        return None


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

    if argv and argv[0] in _SUBCOMMAND_PARSER_BUILDERS:
        sub = get_subcommand_parser(argv[0])
        if sub is not None:
            sys.argv = [sys.argv[0], *argv[1:]]
            argcomplete.autocomplete(sub)
            return True

    parser = argparse.ArgumentParser(prog="mmml")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[c for c in MMML_COMMANDS if c != "completion"],
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
    cmds = " ".join(c for c in MMML_COMMANDS if c != "completion")
    if shell == "bash":
        return f"""# MMML bash completion (top-level commands only; install argcomplete for full flags)
_mmml_completions() {{
  local cur prev opts
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "{cmds}" -- "$cur") )
  fi
}}
complete -F _mmml_completions mmml
"""
    if shell == "zsh":
        zsh_cmds = " ".join(c for c in MMML_COMMANDS if c != "completion")
        return (
            "# MMML zsh completion (top-level commands only)\n"
            "compdef '_arguments \"*:command:((" + zsh_cmds.replace(" ", "\\n") + "))\"' mmml\n"
        )
    if shell == "fish":
        fish_cmds = "\n".join(c for c in MMML_COMMANDS if c != "completion")
        return f"""# MMML fish completion (top-level commands only)
complete -c mmml -f -n '__fish_use_subcommand' -a '{fish_cmds}'
"""
    raise ValueError(f"unsupported shell: {shell}")


def print_shell_completion(shell: str, *, executable: str = "mmml") -> None:
    """Print a shell snippet that enables ``mmml`` tab completion."""
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
    """``mmml completion bash|zsh|fish [--executable NAME]``."""
    parser = argparse.ArgumentParser(
        prog="mmml completion",
        description="Print shell completion script for mmml (pipe into source or eval).",
    )
    parser.add_argument(
        "shell",
        choices=_COMPLETION_SHELLS,
        help="Target shell",
    )
    parser.add_argument(
        "--executable",
        default="mmml",
        help="CLI executable name (default: mmml)",
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
