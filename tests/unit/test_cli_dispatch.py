"""Top-level CLI dispatch, exit-status handling, and JAX_PLATFORMS scrubbing.

``mmml/cli/__main__.py`` is the entry point for all 64 subcommands and one of
the most-edited files in the tree, at 32.5% coverage. Two things here fail
silently:

* **Registry/dispatch drift.** ``validate_command`` accepts anything listed in
  ``_DISPATCH_COMMANDS``, but the actual work is a long ``if/elif`` chain. Add a
  command to the registry and forget the branch, and it falls through to
  ``parser.print_help(); return 1`` -- a usage message, as if the user had
  typo'd, rather than an error naming the missing wiring.

* **Exit status.** ``_hard_exit`` exists because importing PyCHARMM installs a
  Fortran finalizer that resets the process status to 0, so a command returning
  1 reported success to Slurm/CI/Make. That is the same defect that makes the
  live-CHARMM pytest job unable to fail, so the mitigation deserves a test of
  its own -- exercised in a subprocess, since it calls ``os._exit``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from mmml.cli.registry import _DISPATCH_COMMANDS

_MAIN_PY = Path(__file__).resolve().parents[2] / "mmml" / "cli" / "__main__.py"


def _dispatched_commands() -> set[str]:
    """Command literals compared against ``command`` in ``main``'s if/elif chain."""
    tree = ast.parse(_MAIN_PY.read_text(encoding="utf-8"))
    main_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    found: set[str] = set()
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id != "command":
                continue
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    found.add(comp.value)
    return found


# --- registry / dispatch contract -------------------------------------------


def test_every_registered_command_has_a_dispatch_branch():
    """Otherwise the command exists as far as validation is concerned but falls
    through to the generic help text."""
    missing = sorted(set(_DISPATCH_COMMANDS) - _dispatched_commands())
    assert not missing, f"registered but never dispatched: {missing}"


def test_every_dispatch_branch_is_a_registered_command():
    """A branch for an unregistered command is dead code: ``validate_command``
    rejects the name before dispatch is reached."""
    extra = sorted(_dispatched_commands() - set(_DISPATCH_COMMANDS))
    assert not extra, f"dispatched but not registered: {extra}"


def test_the_registry_is_not_empty_and_has_no_duplicates():
    assert len(_DISPATCH_COMMANDS) > 20
    assert len(set(_DISPATCH_COMMANDS)) == len(_DISPATCH_COMMANDS)


def test_command_names_are_shell_safe():
    """These are typed at a shell prompt and used in generated docs."""
    for name in _DISPATCH_COMMANDS:
        assert name == name.strip()
        assert " " not in name
        assert name.replace("-", "").replace("_", "").isalnum(), name


# --- argument handling ------------------------------------------------------


def _run_cli(args: list[str], env_extra: dict[str, str] | None = None):
    """Run ``python -m mmml.cli`` in a subprocess (dispatch calls ``os._exit``)."""
    import os

    env = {**os.environ, "MMML_DISABLE_CHARMM": "1", "MMML_QUIET": "1"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "mmml.cli", *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(_MAIN_PY.parents[2]),
    )


def test_no_command_prints_help_and_succeeds():
    proc = _run_cli([])
    assert proc.returncode == 0, proc.stderr
    assert "mmml" in (proc.stdout + proc.stderr).lower()


def test_unknown_command_fails_with_a_nonzero_status():
    """argparse ``error()`` exits 2; the important part is that it is not 0."""
    proc = _run_cli(["definitely-not-a-command"])
    assert proc.returncode != 0
    assert "definitely-not-a-command" in (proc.stdout + proc.stderr)


def test_help_flag_succeeds():
    proc = _run_cli(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert "command" in (proc.stdout + proc.stderr).lower()


def test_commands_subcommand_lists_registered_commands():
    """``mmml commands`` prints a *grouped* listing; not every registered name
    appears (``doctor`` and ``env`` are referenced in the footer instead), so
    this checks that the grouped body is populated rather than demanding all 64.
    """
    proc = _run_cli(["commands"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr

    listed = [c for c in _DISPATCH_COMMANDS if c in out]
    assert len(listed) >= 20, f"only {len(listed)} of {len(_DISPATCH_COMMANDS)} listed"
    for name in ("md-system", "make-res"):
        assert name in out, f"{name} missing from `mmml commands` output"


# --- exit status ------------------------------------------------------------
#
# _hard_exit calls os._exit for non-zero codes, which terminates the
# interpreter outright -- these must run out of process.


def _hard_exit_status(code: int) -> int:
    proc = subprocess.run(
        [sys.executable, "-c",
         f"from mmml.cli.__main__ import _hard_exit; _hard_exit({code})"],
        capture_output=True, text=True, timeout=300,
        cwd=str(_MAIN_PY.parents[2]),
    )
    return proc.returncode


@pytest.mark.parametrize("code", [1, 2, 3, 42])
def test_nonzero_exit_codes_survive_interpreter_shutdown(code):
    """The whole point: a Fortran atexit must not reset this to 0."""
    assert _hard_exit_status(code) == code


def test_zero_exits_cleanly():
    """Zero takes the normal SystemExit path so MPI_Finalize still runs."""
    assert _hard_exit_status(0) == 0


def test_none_is_treated_as_success():
    proc = subprocess.run(
        [sys.executable, "-c",
         "from mmml.cli.__main__ import _hard_exit; _hard_exit(None)"],
        capture_output=True, text=True, timeout=300,
        cwd=str(_MAIN_PY.parents[2]),
    )
    assert proc.returncode == 0


def test_cli_wrapper_converts_an_exception_into_a_nonzero_status():
    """``cli()`` must not let a traceback escape as exit 0."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import mmml.cli.__main__ as m; m.main = lambda: (_ for _ in ()).throw(RuntimeError('boom')); m.cli()"],
        capture_output=True, text=True, timeout=300,
        cwd=str(_MAIN_PY.parents[2]),
    )
    assert proc.returncode == 1
    assert "boom" in proc.stderr


def test_cli_wrapper_preserves_an_explicit_systemexit_code():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import mmml.cli.__main__ as m; m.main = lambda: (_ for _ in ()).throw(SystemExit(7)); m.cli()"],
        capture_output=True, text=True, timeout=300,
        cwd=str(_MAIN_PY.parents[2]),
    )
    assert proc.returncode == 7


def test_cli_wrapper_passes_through_a_plain_return_code():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import mmml.cli.__main__ as m; m.main = lambda: 3; m.cli()"],
        capture_output=True, text=True, timeout=300,
        cwd=str(_MAIN_PY.parents[2]),
    )
    assert proc.returncode == 3


# --- JAX_PLATFORMS scrubbing ------------------------------------------------
#
# A stale JAX_PLATFORMS=rocm inherited from the shell aborts backend init on an
# NVIDIA node. The import-time block in __main__ strips it before anything
# imports jax; these check the resulting environment rather than the side effect.


def _platforms_after_import(env_extra: dict[str, str]) -> str:
    import os

    env = {**os.environ, "MMML_DISABLE_CHARMM": "1", **env_extra}
    env.pop("JAX_PLATFORMS", None)
    env.update({k: v for k, v in env_extra.items()})
    proc = subprocess.run(
        [sys.executable, "-c",
         "import mmml.cli.__main__; import os; print(os.environ.get('JAX_PLATFORMS', '<unset>'))"],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=str(_MAIN_PY.parents[2]),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_rocm_is_stripped_from_a_mixed_platform_list():
    got = _platforms_after_import({"JAX_PLATFORMS": "rocm,cpu"})
    assert "rocm" not in got
    assert "cpu" in got


def test_a_platform_list_without_rocm_is_left_alone():
    assert _platforms_after_import({"JAX_PLATFORMS": "cpu"}) == "cpu"


def test_rocm_only_falls_back_to_cuda_when_a_gpu_is_visible():
    got = _platforms_after_import(
        {"JAX_PLATFORMS": "rocm", "CUDA_VISIBLE_DEVICES": "0"}
    )
    assert got == "cuda"


def test_rocm_only_is_unset_when_no_gpu_is_visible():
    """Unset means "let JAX auto-select", which is right on a CPU node."""
    got = _platforms_after_import(
        {"JAX_PLATFORMS": "rocm", "CUDA_VISIBLE_DEVICES": "",
         "SLURM_JOB_GPUS": "", "SLURM_JOB_PARTITION": "cpu"}
    )
    assert got == "<unset>"
