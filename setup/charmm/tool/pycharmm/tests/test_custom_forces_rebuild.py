"""Tests that CHARMM notices OpenMM force changes made after the first energy.

CHARMM builds its OpenMM system once and reuses it for later energy and
dynamics commands, copying each stored force into that system as it goes.  A
force created or altered after the system was built is therefore invisible
until the system is rebuilt -- and the failure is silent, because CHARMM
happily reports the old energy.

Every entry point that creates or alters a stored force marks the system
stale so it is rebuilt before the next evaluation.  These tests pin that
down for each flavour of change: adding a force, changing a global
parameter, and adding a particle, through both the wrapper classes and the
pointer handoff.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_custom_forces_rebuild.py -v
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import energy, reset

pytestmark = pytest.mark.requires_feature("OPENMM")


@pytest.fixture(autouse=True)
def _leave_openmm_off():
    """Turn OpenMM back off after each test in this file.

    Every test here enables OpenMM and leaves a force in the store.  Left
    enabled, a later test that expects CHARMM's own dynamics gets OpenMM's
    instead -- which is silent, and shows up somewhere else entirely as a
    puzzling failure (a Langevin run whose velocity capture comes back
    empty, because the OpenMM path never fills it).
    """
    yield
    reset.openmm()

# CHARMM reports energies in kcal/mol; OpenMM energy expressions are in
# kJ/mol.  Every expected value below is derived from the expression rather
# than captured from a run, so the numbers stay meaningful if the platform or
# the arithmetic changes.
_KJ_PER_KCAL = 4.184

# These tests ask whether a contribution is present at all, not what OpenMM's
# arithmetic produces, so the tolerance only needs to be far tighter than a
# missing or stale term while staying loose enough for single-precision
# platforms.
_REL_TOL = 1e-6


def _fresh_openmm_system():
    """Build the one-atom system and turn OpenMM on, with no forces added.

    Each test needs a system with an empty force store, because store
    indices and stored forces both survive until ``omm.clear()``.  Note the
    platform is selected *after* the clear inside
    :func:`setup_single_atom_system`, since clearing resets it to the
    default.
    """
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()


def _total_energy():
    """Return CHARMM's total energy, forcing an OpenMM evaluation first.

    Returns
    -------
    float
        Total energy in kcal/mol.
    """
    energy.show()
    return energy.get_total()


def _require_matching_openmm_package():
    """Skip the caller unless the ``openmm`` package matches CHARMM's OpenMM.

    Returns
    -------
    module
        The imported ``openmm`` module.
    """
    openmm = pytest.importorskip("openmm")
    py_ver = int(openmm.__version__.split(".")[0]) * 10 + \
        int(openmm.__version__.split(".")[1])
    if omm.omm_version() != py_ver:
        c_major, c_minor = divmod(omm.omm_version(), 10)
        pytest.skip(
            f"openmm package is {openmm.__version__} but CHARMM was built "
            f"against OpenMM {c_major}.{c_minor}; forces cannot be passed "
            f"between different OpenMM builds"
        )
    return openmm


def test_wrapper_force_added_after_energy_is_included():
    """A wrapper force added after the first energy must change the energy.

    Regression test: the built OpenMM system was reused unchanged, so the
    force was silently ignored and the energy stayed exactly the same.
    """
    _fresh_openmm_system()
    before = _total_energy()

    force = omm.CustomExternalForce("100.0 + 0*x")
    force.add_particle(0, [])

    after = _total_energy()
    expected = 100.0 / _KJ_PER_KCAL
    assert after - before == pytest.approx(expected, rel=_REL_TOL), (
        "a force added after the first energy evaluation was not included; "
        "CHARMM reused its already-built OpenMM system"
    )


def test_wrapper_global_parameter_change_after_energy_is_applied():
    """Changing a global parameter after the first energy must take effect.

    The stored force is updated correctly; what used to be missing was the
    request to rebuild the OpenMM system, so the old value kept being used.
    """
    _fresh_openmm_system()
    force = omm.CustomExternalForce("k + 0*x")
    force.add_global_parameter("k", 100.0)
    force.add_particle(0, [])

    before = _total_energy()          # k = 100 kJ/mol, already included
    force.set_global_parameter_default_value(0, 300.0)
    after = _total_energy()

    expected = (300.0 - 100.0) / _KJ_PER_KCAL
    assert after - before == pytest.approx(expected, rel=_REL_TOL), (
        "a global-parameter change made after the first energy evaluation "
        "was not applied; CHARMM reused its already-built OpenMM system"
    )


def test_wrapper_particle_added_after_energy_is_included():
    """Adding a particle to an existing force after the first energy applies.

    Covers a per-particle mutator rather than a global parameter: the same
    stale-system failure hid these too.
    """
    _fresh_openmm_system()
    force = omm.CustomExternalForce("100.0 + 0*x")

    before = _total_energy()          # force exists but acts on no particle
    force.add_particle(0, [])
    after = _total_energy()

    expected = 100.0 / _KJ_PER_KCAL
    assert after - before == pytest.approx(expected, rel=_REL_TOL), (
        "a particle added after the first energy evaluation did not "
        "contribute; CHARMM reused its already-built OpenMM system"
    )


def test_force_turned_off_after_energy_stops_contributing():
    """Switching a force off after the first energy must remove its energy.

    The nastier direction of the same bug: the force stayed in the built
    OpenMM system, so a force the user had disabled went on contributing.
    """
    _fresh_openmm_system()
    force = omm.CustomExternalForce("100.0 + 0*x")
    force.add_particle(0, [])

    with_force = _total_energy()
    assert with_force == pytest.approx(100.0 / _KJ_PER_KCAL, rel=_REL_TOL), (
        "test setup is wrong: the force is not contributing to begin with"
    )

    force.turn_off()
    after_off = _total_energy()

    assert after_off == pytest.approx(0.0, abs=1e-8), (
        "a force switched off after the first energy evaluation kept "
        "contributing; CHARMM reused its already-built OpenMM system"
    )


def test_force_turned_back_on_after_energy_contributes_again():
    """Switching a force back on after an energy must restore its energy."""
    _fresh_openmm_system()
    force = omm.CustomExternalForce("100.0 + 0*x")
    force.add_particle(0, [])
    force.turn_off()

    off = _total_energy()
    force.turn_on()
    on = _total_energy()

    expected = 100.0 / _KJ_PER_KCAL
    assert on - off == pytest.approx(expected, rel=_REL_TOL), (
        "a force switched back on after the first energy evaluation did not "
        "contribute again"
    )


def test_pointer_force_added_after_energy_is_included():
    """The pointer handoff must keep rebuilding the system after an add.

    This path already rebuilt correctly; the test guards it against being
    lost while the wrapper path is reworked.
    """
    openmm = _require_matching_openmm_package()
    _fresh_openmm_system()
    before = _total_energy()

    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    omm.add_openmm_force(force)

    after = _total_energy()
    expected = 100.0 / _KJ_PER_KCAL
    assert after - before == pytest.approx(expected, rel=_REL_TOL), (
        "a force handed over by pointer after the first energy evaluation "
        "was not included"
    )
