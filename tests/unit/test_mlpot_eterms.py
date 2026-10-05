"""MLpot energy lands in USER (CHARMM <= c49) or MLPO + MLEL (c52a1)."""

from __future__ import annotations

import math
import types

import pytest

from mmml.interfaces.pycharmmInterface.mlpot import setup as mlpot_setup
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_eterms import (
    MLPOT_ETERM_KEYS,
    mlpot_eterm_kcal_from_terms,
    read_mlpot_eterm_kcal,
)


def _energy_mod(terms: dict[str, float]):
    def get_term_by_name(name: str) -> float:
        if name not in terms:
            raise ValueError(f"{name} is not a valid energy term")
        return terms[name]

    return types.SimpleNamespace(get_term_by_name=get_term_by_name)


def test_keys():
    assert MLPOT_ETERM_KEYS == ("USER", "MLPO", "MLEL")


def test_c49_user_only():
    assert read_mlpot_eterm_kcal(_energy_mod({"USER": -1873.89})) == pytest.approx(-1873.89)


def test_c52a1_mlpo_mlel_user_zero():
    # c52a1 ENER after MLpot registration: USER=0, MLPO=callback return, MLEL=ML/MM elec.
    mod = _energy_mod({"USER": 0.0, "MLPO": -1873.89, "MLEL": -1.5})
    assert read_mlpot_eterm_kcal(mod) == pytest.approx(-1875.39)


def test_c52a1_names_not_yet_assigned():
    # CETERM(MLPO/MLEL) is named lazily on the first MLpot ENER.
    assert read_mlpot_eterm_kcal(_energy_mod({"USER": 0.0})) == 0.0
    assert read_mlpot_eterm_kcal(_energy_mod({})) is None


def test_nonfinite_is_none():
    assert read_mlpot_eterm_kcal(_energy_mod({"MLPO": math.nan})) is None


def test_from_terms_row():
    assert mlpot_eterm_kcal_from_terms({"USER": 0.0, "MLPO": -5.0, "ELEC": 3.0}) == -5.0
    assert mlpot_eterm_kcal_from_terms(None) == 0.0


def test_ml_energy_not_missing_when_only_mlpo():
    terms = {"USER": 0.0, "MLPO": -1873.89, "MLEL": 0.0, "VDW": 0.0, "ELEC": 0.0}
    assert not mlpot_setup._mlpot_ml_energy_missing_in_charmm(0.0, terms, zero_tol_kcalmol=1e-12)
    assert mlpot_setup._effective_mlpot_user_kcal(0.0, terms) == pytest.approx(-1873.89)
    assert mlpot_setup._mlpot_ml_energy_missing_in_charmm(
        0.0, {"USER": 0.0, "VDW": 0.0}, zero_tol_kcalmol=1e-12
    )
