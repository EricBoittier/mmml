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
    mod.pycharmm = mock.Mock()
    mod.pycharmm.lingo.charmm_script = mock.Mock()

    def disable_charmm_domdec():
        if mod._domdec_vacuum_disabled:
            return
        mod.pycharmm.lingo.charmm_script("domdec off")
        mod._domdec_vacuum_disabled = True

    mod.disable_charmm_domdec = disable_charmm_domdec
    return mod


def test_disable_charmm_domdec_runs_once():
    mod = _load_import_pycharmm_stub()
    mod.disable_charmm_domdec()
    mod.disable_charmm_domdec()
    mod.disable_charmm_domdec()
    assert mod.pycharmm.lingo.charmm_script.call_count == 1
    assert mod.pycharmm.lingo.charmm_script.call_args[0][0] == "domdec off"


def test_ensure_vendored_pycharmm_on_path_prefers_repo_root():
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    root = str(repo)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.append("/tmp/fake-charmm-pycharmm-tail")

    from mmml.interfaces.pycharmmInterface.import_pycharmm import _ensure_vendored_pycharmm_on_path

    _ensure_vendored_pycharmm_on_path()
    assert sys.path[0] == root
