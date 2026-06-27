"""CHARMM bomb-level handling during topology/parameter file loads."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call


def _stub_pycharmm(monkeypatch) -> tuple[MagicMock, MagicMock]:
    mock_settings = MagicMock()
    mock_settings.set_bomb_level.side_effect = lambda x: 0
    mock_settings.set_warn_level.side_effect = lambda x: 0
    mock_settings.set_verbosity.side_effect = lambda x: 5
    mock_lingo = MagicMock()
    fake = MagicMock()
    fake.settings = mock_settings
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    monkeypatch.setitem(sys.modules, "pycharmm.settings", mock_settings)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", mock_lingo)
    return fake, mock_settings


def test_charmm_silent_command_restores_prior_levels(monkeypatch):
    mock_settings = MagicMock()
    mock_settings.set_verbosity.side_effect = [5, 0, 5]
    mock_settings.set_warn_level.side_effect = [5, 0, 5]
    mock_settings.set_bomb_level.side_effect = [0, -2, 0]
    fake = MagicMock()
    fake.settings = mock_settings
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    monkeypatch.setitem(sys.modules, "pycharmm.settings", mock_settings)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    with charmm_silent_command():
        pass

    assert mock_settings.set_verbosity.call_args_list == [call(0), call(5)]
    assert mock_settings.set_warn_level.call_args_list == [call(0), call(5)]
    assert mock_settings.set_bomb_level.call_args_list == [call(-2), call(0)]
    fake.lingo.charmm_script.assert_not_called()


def test_charmm_relaxed_bomlev_restores_prior_level(monkeypatch):
    _, mock_settings = _stub_pycharmm(monkeypatch)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        pass

    assert mock_settings.set_bomb_level.call_args_list == [call(-2), call(0)]
    assert mock_settings.set_warn_level.call_args_list == [call(-2), call(0)]


def test_run_charmm_script_quiet_does_not_echo_level_commands(monkeypatch):
    mock_settings = MagicMock()
    mock_settings.set_verbosity.side_effect = [5, 0, 5]
    mock_settings.set_warn_level.side_effect = [5, 0, 5]
    mock_lingo = MagicMock()
    fake = MagicMock()
    fake.settings = mock_settings
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    monkeypatch.setitem(sys.modules, "pycharmm.settings", mock_settings)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", mock_lingo)
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet

    run_charmm_script_quiet("ENER\n")

    mock_lingo.charmm_script.assert_called_once_with("ENER\n")
    assert mock_settings.set_verbosity.call_args_list == [call(0), call(5)]


def test_topology_loaders_do_not_pin_bomlev_zero():
    root = Path(__file__).resolve().parents[2] / "mmml/interfaces/pycharmmInterface"
    for name in ("nbonds_config.py", "mm_energy_forces.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert 'charmm_script("bomlev 0")' not in text
        assert "charmm_script('bomlev 0')" not in text
