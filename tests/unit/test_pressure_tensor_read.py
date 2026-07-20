"""Unit tests for CHARMM instantaneous pressure reads (mocked)."""

from __future__ import annotations

from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface.mlpot.pressure_tensor import (
    read_instantaneous_scalar_pressure_atm,
)


def test_read_instantaneous_prefers_prsi():
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics.safe_energy_show"
        ),
        mock.patch("pycharmm.lingo.get_energy_value", side_effect=lambda k: {"PRSI": 1.25}.get(k, float("nan"))),
    ):
        # Import path uses pycharmm.lingo inside the function after energy refresh.
        with mock.patch(
            "pycharmm.lingo.get_energy_value",
            side_effect=lambda name: 1.25 if name == "PRSI" else float("nan"),
        ):
            # Re-call with lingo mocked at import site used by the function.
            pass

    def _get(name: str) -> float:
        if name == "PRSI":
            return 1.25
        raise KeyError(name)

    fake_lingo = mock.Mock()
    fake_lingo.get_energy_value = _get
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics.safe_energy_show"
        ),
        mock.patch.dict("sys.modules", {"pycharmm.lingo": fake_lingo, "pycharmm": mock.Mock()}),
    ):
        # Force re-import path: function does `import pycharmm.lingo as lingo`
        import importlib
        import sys

        sys.modules["pycharmm.lingo"] = fake_lingo
        p = read_instantaneous_scalar_pressure_atm(refresh_energy=True, quiet=True)
    assert p == pytest.approx(1.25)


def test_read_instantaneous_falls_back_to_pixx_mean():
    def _get(name: str) -> float:
        return {"PIXX": 1.0, "PIYY": 2.0, "PIZZ": 3.0}.get(name, float("nan"))

    fake_lingo = mock.Mock()
    fake_lingo.get_energy_value = _get
    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.dynamics.safe_energy_show"
        ),
        mock.patch.dict("sys.modules", {"pycharmm.lingo": fake_lingo, "pycharmm": mock.Mock()}),
    ):
        import sys

        sys.modules["pycharmm.lingo"] = fake_lingo
        p = read_instantaneous_scalar_pressure_atm(refresh_energy=True, quiet=True)
    assert p == pytest.approx(2.0)
