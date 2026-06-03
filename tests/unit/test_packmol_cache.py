"""Unit tests for Packmol cluster disk cache keys."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.packmol_cache import packmol_cache_key


def test_packmol_cache_key_stable_for_same_inputs():
    kwargs = dict(
        composition=[("DCM", 9)],
        center=(0.0, 0.0, 0.0),
        radius=9.6,
        tolerance=2.0,
        seed=123,
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3,
    )
    assert packmol_cache_key(**kwargs) == packmol_cache_key(**kwargs)


def test_packmol_cache_key_changes_with_composition_or_seed():
    base = dict(
        composition=[("DCM", 9)],
        center=(0.0, 0.0, 0.0),
        radius=9.6,
        tolerance=2.0,
        seed=123,
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3,
    )
    k0 = packmol_cache_key(**base)
    other = {**base, "composition": [("DCM", 10)]}
    assert packmol_cache_key(**other) != k0
    other_seed = {**base, "seed": 124}
    assert packmol_cache_key(**other_seed) != k0
