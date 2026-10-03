"""Focused tests for pyCHARMM DOMDEC crystal safeguards."""

from unittest.mock import patch

import pytest


@pytest.mark.parametrize("crystal_type", ["CUBI", "TETR", "ORTH", "RECT"])
def test_supported_crystals_are_allowed(crystal_type):
    import pycharmm.domdec as domdec

    with patch("pycharmm.crystal.get_crystal_type_direct", return_value=crystal_type):
        domdec._check_crystal_compatibility()


def test_missing_crystal_is_rejected():
    import pycharmm.domdec as domdec

    with patch("pycharmm.crystal.get_crystal_type_direct", return_value=None):
        with pytest.raises(
            domdec.DomdecEngineError, match="requires a defined orthorhombic crystal"
        ):
            domdec._check_crystal_compatibility()


@pytest.mark.parametrize("crystal_type", ["OCTA", "UNKNOWN"])
def test_unsupported_crystal_is_rejected(crystal_type):
    import pycharmm.domdec as domdec

    with patch("pycharmm.crystal.get_crystal_type_direct", return_value=crystal_type):
        with pytest.raises(domdec.DomdecEngineError, match=f"orthorhombic crystal.*{crystal_type}"):
            domdec._check_crystal_compatibility()


@pytest.mark.parametrize(
    "operation, crystal_type",
    [
        ("enable", "RHDO"),
        ("energy", "TRIC"),
    ],
)
def test_guard_runs_before_charmm_command(operation, crystal_type):
    import pycharmm.domdec as domdec

    with (
        patch("pycharmm.crystal.get_crystal_type_direct", return_value=crystal_type),
        patch("pycharmm.domdec.charmm_script") as run_command,
    ):
        with pytest.raises(domdec.DomdecEngineError):
            getattr(domdec, operation)(gpu=False)

    run_command.assert_not_called()


@pytest.mark.parametrize(
    "kwargs, enabled",
    [
        ({"domdec": True}, False),
        ({}, True),
        ({"blade": True}, True),
        ({"omm": True}, True),
    ],
)
def test_dynamics_guard_runs_before_charmm_command(kwargs, enabled):
    import pycharmm.domdec as domdec
    from pycharmm.dynamics import DynamicsScript

    with (
        patch("pycharmm.domdec.is_enabled", return_value=enabled),
        patch("pycharmm.crystal.get_crystal_type_direct", return_value="TRIC"),
        patch("pycharmm.dynamics.script.CommandScript.run") as run_command,
    ):
        with pytest.raises(domdec.DomdecEngineError):
            DynamicsScript(**kwargs).run()

    run_command.assert_not_called()
