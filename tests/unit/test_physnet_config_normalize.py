"""Unit tests for PhysNet checkpoint config key normalization."""

from __future__ import annotations

from mmml.utils.model_checkpoint import (
    canonicalize_physnet_config_for_save,
    normalize_physnet_config,
)


def test_normalize_physnet_config_from_legacy_keys() -> None:
    cfg = {"n_res": 3, "natoms": 34, "features": 32}
    out = normalize_physnet_config(cfg)
    assert out["n_res"] == 3
    assert out["n_refinement_blocks"] == 3
    assert out["natoms"] == 34
    assert out["max_padded_atoms"] == 34
    assert out["features"] == 32


def test_normalize_physnet_config_from_canonical_keys() -> None:
    cfg = {"n_refinement_blocks": 2, "max_padded_atoms": 60}
    out = normalize_physnet_config(cfg)
    assert out["n_res"] == 2
    assert out["n_refinement_blocks"] == 2
    assert out["natoms"] == 60
    assert out["max_padded_atoms"] == 60


def test_canonicalize_for_save_matches_normalize() -> None:
    legacy = {"n_res": 5, "natoms": 20}
    assert canonicalize_physnet_config_for_save(legacy) == normalize_physnet_config(legacy)
