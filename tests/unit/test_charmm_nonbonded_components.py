"""CHARMM nonbonded component mapping (PBC image terms)."""

from __future__ import annotations

from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
    _charmm_nb_term_sum,
    charmm_nonbonded_energy_components_kcalmol,
)


def test_charmm_nb_term_sum_missing_terms_are_zero(monkeypatch) -> None:
    def _fake(term: str):
        return {"VDW": 1.5, "IMNB": 0.25}.get(term)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.charmm_bonded_term_kcalmol",
        _fake,
    )
    assert _charmm_nb_term_sum("VDW", "IMNB", "MISSING") == 1.75


def test_charmm_nonbonded_components_include_image_terms(monkeypatch) -> None:
    values = {"VDW": 1.0, "IMNB": 0.5, "ELEC": 0.8, "IMEL": -0.3, "EXTE": 0.0}

    def _fake(term: str):
        return values.get(term)

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.charmm_bonded_term_kcalmol",
        _fake,
    )
    out = charmm_nonbonded_energy_components_kcalmol()
    assert out["vdw"] == 1.5
    assert out["elec"] == 0.5
    assert out["total"] == 2.0
