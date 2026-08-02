"""A failing pytest session must report a failing exit status, even with CHARMM.

Importing pycharmm installs a Fortran/MPI finalizer that runs during interpreter
shutdown and resets the process exit status to 0. Before the ``pytest_unconfigure``
hook in ``tests/conftest.py``, this held on any machine with a CHARMM build::

    pytest <test that loads pycharmm and then fails>; echo $?   ->  0

Everything that trusts an exit code -- Slurm, Make, CI, the validation campaign,
``run_pycharmm_smoke_pytest.sh``'s ``|| status=1`` -- was blind to failures in the
live suite. This file guards the fix in both directions: failures must surface,
and clean runs must still shut down normally so OpenMPI can finalize.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tests import conftest as mmml_conftest

_TESTS_UNIT = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_UNIT.parents[1]


# --- the recording half (safe to exercise in-process) -----------------------


def test_a_failing_session_is_recorded():
    mmml_conftest._FORCED_EXIT_STATUS.clear()
    try:
        mmml_conftest.pytest_sessionfinish(session=None, exitstatus=1)
        assert mmml_conftest._FORCED_EXIT_STATUS.get("code") == 1
    finally:
        mmml_conftest._FORCED_EXIT_STATUS.clear()


def test_a_clean_session_records_nothing():
    mmml_conftest._FORCED_EXIT_STATUS.clear()
    try:
        mmml_conftest.pytest_sessionfinish(session=None, exitstatus=0)
        assert "code" not in mmml_conftest._FORCED_EXIT_STATUS
    finally:
        mmml_conftest._FORCED_EXIT_STATUS.clear()


def test_a_non_integer_status_is_treated_as_failure():
    mmml_conftest._FORCED_EXIT_STATUS.clear()
    try:
        mmml_conftest.pytest_sessionfinish(session=None, exitstatus="boom")
        assert mmml_conftest._FORCED_EXIT_STATUS.get("code") == 1
    finally:
        mmml_conftest._FORCED_EXIT_STATUS.clear()


def test_unconfigure_is_inert_without_charmm():
    """Under MMML_DISABLE_CHARMM (and in CI) pycharmm is never imported, so the
    hook must return rather than kill the interpreter -- if this were wrong the
    test session running it would die here."""
    mmml_conftest._FORCED_EXIT_STATUS.clear()
    mmml_conftest._FORCED_EXIT_STATUS["code"] = 1
    try:
        assert not mmml_conftest._pycharmm_was_loaded()
        mmml_conftest.pytest_unconfigure(config=None)  # must simply return
    finally:
        mmml_conftest._FORCED_EXIT_STATUS.clear()


def test_the_escape_hatch_disables_the_hook(monkeypatch):
    """MMML_NO_FORCE_PYTEST_EXIT exists for debugging a shutdown problem."""
    monkeypatch.setenv("MMML_NO_FORCE_PYTEST_EXIT", "1")
    monkeypatch.setattr(mmml_conftest, "_pycharmm_was_loaded", lambda: True)
    mmml_conftest._FORCED_EXIT_STATUS.clear()
    mmml_conftest._FORCED_EXIT_STATUS["code"] = 1
    try:
        mmml_conftest.pytest_unconfigure(config=None)  # must not os._exit
    finally:
        mmml_conftest._FORCED_EXIT_STATUS.clear()


def test_loaded_detection_matches_sys_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycharmm.energy", object())
    assert mmml_conftest._pycharmm_was_loaded()


# --- end to end (the part that actually calls os._exit) ---------------------


def _charmm_available() -> bool:
    return mmml_conftest.can_import_pycharmm()


def _run_probe(body: str) -> int:
    """Write a throwaway test under tests/unit/ and report pytest's exit status.

    It has to live inside ``tests/`` for ``tests/conftest.py`` -- which owns the
    hook under test -- to be collected at all.
    """
    probe = _TESTS_UNIT / f"test_zz_exitprobe_{uuid.uuid4().hex[:8]}.py"
    probe.write_text(body, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q",
             "-p", "no:cacheprovider", "--no-header"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "MMML_QUIET": "1"},
        )
        return proc.returncode
    finally:
        probe.unlink(missing_ok=True)


_LOAD_CHARMM = (
    "from mmml.interfaces.pycharmmInterface.import_pycharmm import "
    "ensure_pycharmm_loaded\n"
)


@pytest.mark.pycharmm
@pytest.mark.skipif(not _charmm_available(), reason="needs a real libcharmm")
def test_failing_charmm_session_exits_nonzero():
    """The defect this hook exists for: without it this returns 0."""
    code = _run_probe(
        f"def test_fail():\n"
        f"    {_LOAD_CHARMM.strip()}\n"
        f"    ensure_pycharmm_loaded()\n"
        f"    assert 1 == 2\n"
    )
    assert code != 0, "CHARMM shutdown masked a failing pytest session"


@pytest.mark.pycharmm
@pytest.mark.skipif(not _charmm_available(), reason="needs a real libcharmm")
def test_passing_charmm_session_exits_zero():
    """Clean runs must take the normal shutdown path so MPI_Finalize runs."""
    code = _run_probe(
        f"def test_pass():\n"
        f"    {_LOAD_CHARMM.strip()}\n"
        f"    ensure_pycharmm_loaded()\n"
        f"    assert True\n"
    )
    assert code == 0


def test_failing_session_without_charmm_exits_nonzero():
    """Baseline: plain pytest semantics are untouched."""
    assert _run_probe("def test_fail():\n    assert 1 == 2\n") != 0
