"""Unit tests for CHARMM instantaneous pressure resolution."""

from __future__ import annotations

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
    scalar_pressure_atm_from_energy_getters,
)


def test_scalar_pressure_prefers_prsi():
    def _get(name: str) -> float:
        return 1.25 if name == "PRSI" else float("nan")

    assert scalar_pressure_atm_from_energy_getters(_get) == pytest.approx(1.25)


def test_scalar_pressure_falls_back_to_pixx_mean():
    def _get(name: str) -> float:
        if name == "PRSI":
            return float("nan")
        return {"PIXX": 1.0, "PIYY": 2.0, "PIZZ": 3.0}[name]

    assert scalar_pressure_atm_from_energy_getters(_get) == pytest.approx(2.0)


def test_scalar_pressure_raises_when_unavailable():
    def _get(_name: str) -> float:
        return float("nan")

    with pytest.raises(RuntimeError, match="unavailable"):
        scalar_pressure_atm_from_energy_getters(_get)
