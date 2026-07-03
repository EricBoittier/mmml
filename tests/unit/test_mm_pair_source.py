"""Unit tests for decomposed MLpot MM pair source resolution."""

from __future__ import annotations

import argparse

import pytest


def test_resolve_mm_pair_source_defaults_to_jax() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import resolve_mm_pair_source

    assert resolve_mm_pair_source() == "jax"
    assert resolve_mm_pair_source(argparse.Namespace()) == "jax"


def test_resolve_mm_pair_source_cli_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import resolve_mm_pair_source

    ns = argparse.Namespace(mm_pair_source="charmm_callback")
    assert resolve_mm_pair_source(ns) == "charmm_callback"
    monkeypatch.setenv("MMML_MM_PAIR_SOURCE", "charmm_callback")
    assert resolve_mm_pair_source() == "charmm_callback"
    monkeypatch.setenv("MMML_MM_PAIR_SOURCE", "jax")
    assert resolve_mm_pair_source(ns) == "charmm_callback"
