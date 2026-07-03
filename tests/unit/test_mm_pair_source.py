"""Unit tests for decomposed MLpot MM pair source resolution."""

from __future__ import annotations

import argparse

import pytest


def test_resolve_mm_pair_source_defaults_to_charmm_callback() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import resolve_mm_pair_source

    assert resolve_mm_pair_source() == "charmm_callback"
    assert resolve_mm_pair_source(argparse.Namespace()) == "charmm_callback"


def test_resolve_mm_pair_source_jax_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import resolve_mm_pair_source

    assert resolve_mm_pair_source(argparse.Namespace(mm_pair_source="jax")) == "jax"
    monkeypatch.setenv("MMML_MM_PAIR_SOURCE", "jax")
    assert resolve_mm_pair_source() == "jax"
    monkeypatch.delenv("MMML_MM_PAIR_SOURCE", raising=False)
    assert resolve_mm_pair_source() == "charmm_callback"
