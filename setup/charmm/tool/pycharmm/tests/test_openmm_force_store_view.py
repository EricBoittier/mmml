"""Tests for looking at what CHARMM's OpenMM force store holds.

Until `get_forces()` there was no way to ask. You could not count the stored
forces, and the only way to learn whether one was switched on was to toggle
it and read back the previous value -- which changed the thing you were
asking about.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_force_store_view.py -v
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import reset

pytestmark = pytest.mark.requires_feature("OPENMM")


@pytest.fixture(autouse=True)
def _fresh_store():
    """One-atom system with an empty store; OpenMM left off afterwards."""
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    yield
    reset.openmm()


def test_empty_store_reports_no_forces():
    """An empty store is an empty table, not an error or a guess."""
    forces = omm.get_forces()
    assert forces.empty
    assert list(forces.columns) == [
        "index", "kind", "eterm", "eterm_is_default", "enabled"]


def test_each_added_force_appears_once_in_index_order():
    """The obvious question -- what did I add? -- answered in order."""
    omm.CustomExternalForce("0*x")
    omm.CustomBondForce("r")

    forces = omm.get_forces()
    assert list(forces["index"]) == [0, 1]
    assert list(forces["kind"]) == [omm.CustomForceType.EXTERNAL,
                                    omm.CustomForceType.BOND]


def test_default_energy_term_is_reported_as_default():
    """A force nobody pinned reports its class default, flagged as such."""
    omm.CustomExternalForce("0*x")

    row = omm.get_forces().iloc[0]
    assert row["eterm"] is omm.EtermBucket.CFEX
    assert row["eterm_is_default"]


def test_explicit_energy_term_is_distinguished_from_the_default():
    """"Pinned to CFEX" and "defaults to CFEX" must not look identical.

    Telling them apart is the point: otherwise a script that sets terms
    conditionally gives no way to see whether its setting took.
    """
    force = omm.CustomExternalForce("0*x")
    force.set_eterm(omm.EtermBucket.CFCV)

    row = omm.get_forces().iloc[0]
    assert row["eterm"] is omm.EtermBucket.CFCV
    assert not row["eterm_is_default"]


def test_disabled_force_is_visibly_disabled():
    """A switched-off force contributes nothing, so it must be visible.

    This is the state that was hardest to see before: reading it meant
    toggling the force, which changed it.
    """
    force = omm.CustomExternalForce("0*x")
    assert omm.get_forces().iloc[0]["enabled"]

    force.turn_off()
    assert not omm.get_forces().iloc[0]["enabled"]

    force.turn_on()
    assert omm.get_forces().iloc[0]["enabled"]


def test_reading_the_store_does_not_disturb_it():
    """Looking must not change what is being looked at."""
    force = omm.CustomExternalForce("0*x")
    force.turn_off()

    before = omm.get_forces()
    for _ in range(3):
        omm.get_forces()
    after = omm.get_forces()

    assert before.equals(after)
    assert not after.iloc[0]["enabled"], "reading flipped the enabled flag"


def test_indices_match_what_add_openmm_force_returned():
    """The index in the table is the one the caller was handed."""
    openmm = pytest.importorskip("openmm")
    py_ver = (int(openmm.__version__.split(".")[0]) * 10
              + int(openmm.__version__.split(".")[1]))
    if omm.omm_version() != py_ver:
        pytest.skip("openmm package does not match CHARMM's OpenMM")

    f = openmm.CustomExternalForce("0*x")
    f.addParticle(0, [])
    index = omm.add_openmm_force(f)

    assert index in list(omm.get_forces()["index"])


def test_out_of_range_kind_is_reported_not_fatal():
    """A bad store index must come back as "no such force", not kill the run.

    The C++ side indexes with std::vector::at, which throws. Reached across
    the C interface from Fortran there is no handler, so the exception would
    hit std::terminate and take the whole process down rather than report a
    bad index. This test only passes if that is caught.
    """
    import ctypes

    omm.CustomExternalForce("0*x")          # one force, so index 0 is valid

    lib = omm.lib
    lib.api_cf_get_kind.argtypes = [ctypes.c_int]
    lib.api_cf_get_kind.restype = ctypes.c_int

    assert lib.api_cf_get_kind(ctypes.c_int(0)) >= 0
    assert lib.api_cf_get_kind(ctypes.c_int(999)) == -1
    assert lib.api_cf_get_kind(ctypes.c_int(-1)) == -1


def test_show_forces_prints_a_row_per_force(capsys):
    """show_forces() is for looking at; check it says the useful things."""
    force = omm.CustomExternalForce("0*x")
    force.set_eterm(omm.EtermBucket.CFCV)
    omm.CustomBondForce("r")
    omm.force_turn_off(1)

    omm.show_forces()
    printed = capsys.readouterr().out

    assert "EXTERNAL" in printed
    assert "BOND" in printed
    assert "CFCV" in printed
    assert "off: not in the system" in printed, printed
    assert "set explicitly" in printed, printed


def test_show_forces_says_so_when_the_store_is_empty(capsys):
    """Silence would read as "it printed nothing", not "there is nothing"."""
    omm.show_forces()
    assert "empty" in capsys.readouterr().out
