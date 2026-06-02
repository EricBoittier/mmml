"""CHARMM bomb-level handling during topology/parameter file loads."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock


def test_read_cgenff_toppar_does_not_force_bomlev_zero(monkeypatch):
    scripts: list[str] = []

    mock_read = MagicMock()
    mock_settings = MagicMock()
    mock_settings.set_bomb_level.side_effect = [0, -2, 0]
    mock_settings.set_warn_level.side_effect = [0, -2, 0]
    mock_lingo = MagicMock()
    mock_lingo.charmm_script.side_effect = lambda s: scripts.append(s)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nbonds_config._rtf_path_without_drude_autogen",
        lambda p: str(p),
    )
    monkeypatch.setattr("pycharmm.read", mock_read)
    monkeypatch.setattr("pycharmm.settings", mock_settings)
    monkeypatch.setattr("pycharmm.lingo", mock_lingo)

    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    read_cgenff_toppar()

    assert mock_read.rtf.called
    assert mock_read.prm.called
    assert mock_settings.set_bomb_level.call_args_list[-1][0][0] == 0
    assert not any("bomlev 0" in s.lower() for s in scripts)


def test_charmm_relaxed_bomlev_restores_prior_level(monkeypatch):
    mock_settings = MagicMock()
    mock_settings.set_bomb_level.side_effect = [0, -2, 0]
    mock_settings.set_warn_level.side_effect = [0, -2, 0]
    scripts: list[str] = []
    mock_lingo = MagicMock()
    mock_lingo.charmm_script.side_effect = lambda s: scripts.append(s)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.settings",
        mock_settings,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.pycharmm.lingo",
        mock_lingo,
    )

    from mmml.interfaces.pycharmmInterface.import_pycharmm import charmm_relaxed_bomlev

    with charmm_relaxed_bomlev():
        pass

    assert mock_settings.set_bomb_level.call_args_list[-1][0][0] == 0
    assert scripts[0].startswith("bomlev -2")
    assert "bomlev 0" in scripts[-1]
