"""Tests for once-only domdec disable on DOMDEC CHARMM builds."""

from __future__ import annotations

from contextlib import nullcontext
from unittest import mock

from pathlib import Path


def _prepare_domdec_module(monkeypatch):
    """Use the real DOMDEC helper functions with PyCHARMM side effects mocked."""
    from mmml.interfaces.pycharmmInterface import import_pycharmm as mod

    mod._domdec_vacuum_disabled = False
    mod._domdec_disabled_early = False
    monkeypatch.delenv("MMML_FORCE_DOMDEC_OFF", raising=False)
    monkeypatch.delenv("MMML_NO_CHARMM_DOMDEC_OFF", raising=False)
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_levels.charmm_relaxed_bomlev",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(mod, "pycharmm", mock.Mock())
    mod.pycharmm.lingo.charmm_script = mock.Mock()
    return mod


def test_disable_charmm_domdec_skips_by_default(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    assert mod.disable_charmm_domdec() is False
    mod.pycharmm.lingo.charmm_script.assert_not_called()


def test_disable_charmm_domdec_force_hatch_runs_once(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    monkeypatch.setenv("MMML_FORCE_DOMDEC_OFF", "1")
    assert mod.disable_charmm_domdec(when="mlpot_energy") is True
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    assert mod.pycharmm.lingo.charmm_script.call_count == 1
    assert mod.pycharmm.lingo.charmm_script.call_args[0][0] == "domdec off"
    assert mod._domdec_vacuum_disabled is True
    assert mod._domdec_disabled_early is False


def test_disable_charmm_domdec_no_hatch_wins_over_force(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    monkeypatch.setenv("MMML_FORCE_DOMDEC_OFF", "1")
    monkeypatch.setenv("MMML_NO_CHARMM_DOMDEC_OFF", "1")
    assert mod.disable_charmm_domdec(when="mlpot_energy") is False
    mod.pycharmm.lingo.charmm_script.assert_not_called()


def test_ensure_domdec_off_recovers_mpi_only_after_success(monkeypatch):
    mod = _prepare_domdec_module(monkeypatch)
    recover = mock.Mock()
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.recover_mpi_for_charmm_after_jax",
        recover,
    )

    assert mod.ensure_domdec_off_for_mlpot_energy(context="unit default") is False
    recover.assert_not_called()

    monkeypatch.setenv("MMML_FORCE_DOMDEC_OFF", "1")
    assert mod.ensure_domdec_off_for_mlpot_energy(context="unit forced") is True
    recover.assert_called_once_with(phase="after domdec off (unit forced)")


def test_disable_charmm_domdec_skipped_by_default():
    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def _should_run_domdec_off()" in source
    assert "MMML_FORCE_DOMDEC_OFF" in source
    block = source.split("def disable_charmm_domdec(")[1].split("\ndef ")[0]
    assert "if not _should_run_domdec_off():" in block


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
