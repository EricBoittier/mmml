"""Canonical calculator path registry."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.canonical_paths import CANONICAL


def test_canonical_paths_registry() -> None:
    assert "setup_calculator" in CANONICAL["hybrid_calculator_factory"]
    assert len(CANONICAL) >= 4
