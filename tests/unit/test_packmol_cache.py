"""Unit tests for Packmol cluster disk cache keys."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.packmol_cache import (
    CACHE_VERSION,
    packmol_cache_fingerprint,
    packmol_cache_key,
    packmol_prep_settings_from_mapping,
)


def _base_kwargs() -> dict:
    return dict(
        composition=[("DCM", 9)],
        placement="cube",
        center=(0.0, 0.0, 0.0),
        cube_side=38.0,
        radius=None,
        tolerance=2.0,
        seed=123,
        charmm_sd_steps=50,
        charmm_abnr_steps=100,
        charmm_tolenr=1e-3,
        charmm_tolgrd=1e-3,
    )


def test_packmol_cache_key_stable_for_same_inputs():
    kwargs = _base_kwargs()
    assert packmol_cache_key(**kwargs) == packmol_cache_key(**kwargs)


def test_packmol_cache_key_changes_with_composition_or_seed():
    base = _base_kwargs()
    k0 = packmol_cache_key(**base)
    other = {**base, "composition": [("DCM", 10)]}
    assert packmol_cache_key(**other) != k0
    other_seed = {**base, "seed": 124}
    assert packmol_cache_key(**other_seed) != k0
    other_placement = {**base, "placement": "sphere", "radius": 19.0, "cube_side": None}
    assert packmol_cache_key(**other_placement) != k0


def test_packmol_cache_key_changes_with_padding_tolerance_spacing():
    base = _base_kwargs()
    k0 = packmol_cache_key(**base)
    assert packmol_cache_key(**{**base, "packmol_padding_A": 2.0}) != k0
    assert packmol_cache_key(**{**base, "tolerance": 3.0}) != k0
    assert packmol_cache_key(**{**base, "spacing": 7.0}) != k0


def test_packmol_cache_key_changes_with_prep_gate_settings():
    base = _base_kwargs()
    k0 = packmol_cache_key(**base)
    gates = packmol_prep_settings_from_mapping({"max_grms_before_dyn": 40.0})
    assert packmol_cache_key(**{**base, "prep_gate_settings": gates}) != k0


def test_packmol_cache_fingerprint_includes_version_and_gates():
    fp = packmol_cache_fingerprint(
        **_base_kwargs(),
        packmol_padding_A=1.5,
        spacing=5.0,
        prep_gate_settings={"max_grms_before_dyn": 50.0, "no_scale_max_grms": True},
    )
    assert fp["version"] == CACHE_VERSION
    assert fp["packmol_padding_A"] == 1.5
    assert fp["spacing"] == 5.0
    assert fp["prep_gate_settings"] == {
        "max_grms_before_dyn": 50.0,
        "no_scale_max_grms": True,
    }
