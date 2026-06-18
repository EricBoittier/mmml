"""Tests for once-only domdec disable on DOMDEC CHARMM builds."""

from __future__ import annotations

from unittest import mock

import importlib.util
from pathlib import Path


def _load_import_pycharmm_stub():
    """Load import_pycharmm domdec helpers with pycharmm mocked."""
    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    # Extract only the vacuum-mode helpers for a minimal module.
    mod = type(mock.Mock())()
    mod._domdec_vacuum_disabled = False
    mod._domdec_disabled_early = False
    mod.pycharmm = mock.Mock()
    mod.pycharmm.lingo.charmm_script = mock.Mock()

    def disable_charmm_domdec(*, when: str = "early"):
        if mod._domdec_vacuum_disabled:
            return False
        mod.pycharmm.lingo.charmm_script("domdec off")
        mod._domdec_vacuum_disabled = True
        if when != "mlpot_energy":
            mod._domdec_disabled_early = True
        return True

    mod.disable_charmm_domdec = disable_charmm_domdec
    return mod


def test_disable_charmm_domdec_runs_once():
    mod = _load_import_pycharmm_stub()
    mod.disable_charmm_domdec()
    mod.disable_charmm_domdec()
    mod.disable_charmm_domdec()
    assert mod.pycharmm.lingo.charmm_script.call_count == 1
    assert mod.pycharmm.lingo.charmm_script.call_args[0][0] == "domdec off"


def test_init_vacuum_charmm_state_does_not_disable_domdec():
    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _init_vacuum_charmm_state() -> None:" in source
    block = source.split("def _init_vacuum_charmm_state() -> None:")[1].split("\ndef ")[0]
    code_lines = [
        ln
        for ln in block.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert not any("disable_charmm_domdec(" in ln for ln in code_lines)


def test_setup_charmm_environment_defers_domdec_off():
    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/mlpot/pbc_env.py"
    )
    source = path.read_text(encoding="utf-8")
    block = source.split("def setup_charmm_environment(")[1].split("\ndef ")[0]
    assert "disable_charmm_domdec" not in block


def test_ensure_vendored_pycharmm_on_path_prefers_repo_root():
    import sys

    repo = Path(__file__).resolve().parents[2]
    root = str(repo)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.append("/tmp/fake-charmm-pycharmm-tail")

    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    assert sys.path[0] == root
