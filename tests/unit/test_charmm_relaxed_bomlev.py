"""CHARMM bomb-level handling during topology/parameter file loads."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


def _stub_pycharmm(monkeypatch) -> tuple[MagicMock, list[str]]:
    scripts: list[str] = []
    mock_settings = MagicMock()
    mock_settings.set_bomb_level.side_effect = lambda x: 0
    mock_settings.set_warn_level.side_effect = lambda x: 0
    mock_settings.set_verbosity.side_effect = lambda x: 5
    mock_lingo = MagicMock()
    mock_lingo.charmm_script.side_effect = lambda s: scripts.append(s)
    fake = MagicMock()
    fake.settings = mock_settings
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    monkeypatch.setitem(sys.modules, "pycharmm.settings", mock_settings)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", mock_lingo)
    return fake, scripts


def test_charmm_silent_command_restores_prior_levels(monkeypatch):
    scripts: list[str] = []
    mock_settings = MagicMock()
    mock_settings.set_verbosity.side_effect = [5, 0, 5]
    mock_settings.set_warn_level.side_effect = [5, 0, 5]
    mock_settings.set_bomb_level.side_effect = [0, -2, 0]
    mock_lingo = MagicMock()
    mock_lingo.charmm_script.side_effect = lambda s: scripts.append(s)
    fake = MagicMock()
    fake.settings = mock_settings
    fake.lingo = mock_lingo
    monkeypatch.setitem(sys.modules, "pycharmm", fake)
    monkeypatch.setitem(sys.modules, "pycharmm.settings", mock_settings)
    monkeypatch.setitem(sys.modules, "pycharmm.lingo", mock_lingo)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    with charmm_silent_command():
        pass

    assert "PRNLev 0" in scripts[0]
    assert "bomlev -2" in scripts[0]
    assert scripts[-1].startswith("PRNLev 5")


def test_charmm_relaxed_bomlev_restores_prior_level(monkeypatch):
    _, scripts = _stub_pycharmm(monkeypatch)
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        pass

    assert any("bomlev -2" in s for s in scripts)
    assert any(s.startswith("bomlev 0") for s in scripts)


def test_topology_loaders_do_not_pin_bomlev_zero():
    root = Path(__file__).resolve().parents[2] / "mmml/interfaces/pycharmmInterface"
    for name in ("nbonds_config.py", "mm_energy_forces.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert 'charmm_script("bomlev 0")' not in text
        assert "charmm_script('bomlev 0')" not in text
