"""CHARMM nonbonded component mapping (PBC image terms)."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    _charmm_active_energy_terms,
    _charmm_nb_term_sum,
    charmm_nonbonded_energy_components_kcalmol,
)


def test_charmm_nb_term_sum_missing_terms_are_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.cgenff_bonded_reference._charmm_active_energy_terms",
        lambda: {"VDW": 1.5, "IMNB": 0.25},
    )
    assert _charmm_nb_term_sum("VDW", "IMNB", "MISSING") == 1.75


def test_charmm_nonbonded_components_include_image_terms(monkeypatch) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.cgenff_bonded_reference._charmm_active_energy_terms",
        lambda: {
            "VDW": 1.0,
            "IMNB": 0.5,
            "ELEC": 0.8,
            "IMEL": -0.3,
            "EXTE": 0.0,
        },
    )
    out = charmm_nonbonded_energy_components_kcalmol()
    assert out["vdw"] == 1.5
    assert out["elec"] == 0.5
    assert out["total"] == 2.0
