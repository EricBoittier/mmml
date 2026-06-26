"""Unit tests for MM neighbor-list backend selection."""

from __future__ import annotations

import pytest


def test_pick_static_rebuild_backend_auto_prefers_vesin(monkeypatch):
    from mmml.interfaces.pycharmmInterface.nl_backend import pick_static_rebuild_backend

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_vesin",
        lambda: True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_jax_md",
        lambda: True,
    )
    assert pick_static_rebuild_backend("auto") == "vesin"


def test_pick_static_rebuild_backend_auto_falls_back_to_jax_md_without_vesin(monkeypatch):
    from mmml.interfaces.pycharmmInterface.nl_backend import pick_static_rebuild_backend

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_vesin",
        lambda: False,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_jax_md",
        lambda: True,
    )
    assert pick_static_rebuild_backend("auto") == "jax_md"


def test_pick_static_rebuild_backend_explicit_jax_md(monkeypatch):
    from mmml.interfaces.pycharmmInterface.nl_backend import pick_static_rebuild_backend

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_vesin",
        lambda: True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.nl_backend.have_jax_md",
        lambda: True,
    )
    assert pick_static_rebuild_backend("jax_md") == "jax_md"


@pytest.mark.parametrize("env_value", ["vesin", "jax_md", "cell_list"])
def test_resolve_mm_nl_backend_env(monkeypatch, env_value):
    from mmml.interfaces.pycharmmInterface.nl_backend import resolve_mm_nl_backend

    monkeypatch.setenv("MMML_MM_NL_BACKEND", env_value)
    assert resolve_mm_nl_backend() == env_value
