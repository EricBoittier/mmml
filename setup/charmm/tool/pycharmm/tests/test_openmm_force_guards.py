"""Tests for the guidance pyCHARMM gives when handing over an OpenMM force.

Three things here, all about not being silent:

* A force CHARMM already builds for itself is refused permanently, and the
  refusal says what would be double-counted and where to set it instead.
* A force group set on the force is reported, because CHARMM overwrites it.
* `system_changed()` makes CHARMM notice a force edited through the caller's
  own reference, which is the one change it cannot see for itself.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_force_guards.py -v
"""

import warnings

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import energy, reset

openmm = pytest.importorskip("openmm")

_PY_VER = (int(openmm.__version__.split(".")[0]) * 10
           + int(openmm.__version__.split(".")[1]))
if omm.omm_version() != _PY_VER:
    _c_major, _c_minor = divmod(omm.omm_version(), 10)
    pytest.skip(
        f"openmm package is {openmm.__version__} but CHARMM was built "
        f"against OpenMM {_c_major}.{_c_minor}; forces cannot be passed "
        f"between different OpenMM builds",
        allow_module_level=True,
    )

pytestmark = pytest.mark.requires_feature("OPENMM")

_KJ_PER_KCAL = 4.184
_REL_TOL = 1e-6


@pytest.fixture(autouse=True)
def _fresh_openmm_system():
    """One-atom system with an empty store; OpenMM left off afterwards."""
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    yield
    reset.openmm()


def _total_energy():
    energy.show()
    return energy.get_total()


# --------------------------------------------------------------------------
# Refusing forces CHARMM already builds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("class_name, expect_phrase", [
    ("NonbondedForce", "nonbonded interactions"),
    ("HarmonicBondForce", "bond stretching"),
    ("HarmonicAngleForce", "angle bending"),
    ("PeriodicTorsionForce", "dihedral torsions"),
    ("CMMotionRemover", "centre-of-mass motion removal"),
])
def test_forces_charmm_owns_are_refused_with_a_reason(class_name,
                                                      expect_phrase):
    """The refusal must name what would be counted twice."""
    cls = getattr(openmm, class_name, None)
    if cls is None:                                  # pragma: no cover
        pytest.skip(f"this openmm build has no {class_name}")
    with pytest.raises(TypeError) as exc:
        omm.add_openmm_force(cls())
    message = str(exc.value)
    assert expect_phrase in message, message
    assert "twice" in message, message


def test_refusal_does_not_promise_future_support():
    """These cannot ever be supported, so the message must not say "yet"."""
    with pytest.raises(TypeError) as exc:
        omm.add_openmm_force(openmm.NonbondedForce())
    assert "yet" not in str(exc.value).lower(), str(exc.value)


def test_refusal_points_at_the_charmm_setting():
    """A caller told "no" needs to know where to set it instead."""
    with pytest.raises(TypeError, match="NBONds"):
        omm.add_openmm_force(openmm.NonbondedForce())


def test_genuinely_unsupported_class_still_says_yet():
    """A force that does not collide with CHARMM's terms may come later.

    The two refusals must stay distinguishable: "we might add this" for a
    class that is merely unwired, versus "this can never work" for one that
    would double-count. Every real OpenMM force is in one list or the other,
    so the helper is called directly with a name in neither.
    """
    with pytest.raises(TypeError) as exc:
        omm._refuse_unsupported_class("SomeFutureForce")
    assert "yet" in str(exc.value)
    assert "twice" not in str(exc.value)


# --------------------------------------------------------------------------
# Force groups
# --------------------------------------------------------------------------

def test_force_group_set_by_the_user_is_reported():
    """CHARMM overwrites the group, so accepting it silently misleads."""
    force = openmm.CustomExternalForce("0*x")
    force.addParticle(0, [])
    force.setForceGroup(7)
    with pytest.warns(RuntimeWarning, match="force group 7"):
        omm.add_openmm_force(force)


def test_default_force_group_is_not_reported():
    """Group 0 carries no intent, so warning about it would be noise."""
    force = openmm.CustomExternalForce("0*x")
    force.addParticle(0, [])
    # pytest.warns(None) is gone in modern pytest, and "assert no warning of
    # this kind" is what is wanted here rather than "assert no warning at all".
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        omm.add_openmm_force(force)
    groups = [w for w in record if "force group" in str(w.message)]
    assert not groups, [str(w.message) for w in groups]


def test_custom_nonbonded_force_is_accepted_with_a_caveat():
    """It works, but it is additive and fussy about exclusions -- say so."""
    force = openmm.CustomNonbondedForce("0")
    with pytest.warns(RuntimeWarning, match="on top of CHARMM's own"):
        try:
            omm.add_openmm_force(force)
        except RuntimeError:
            # CHARMM may still refuse it for particle-count reasons; the
            # warning is what this test is about, and it comes first.
            pass


# --------------------------------------------------------------------------
# system_changed()
# --------------------------------------------------------------------------

def test_edit_through_the_callers_own_reference_needs_system_changed():
    """The case CHARMM cannot see: the caller edits the force it handed over.

    Both halves matter. Without the call the old energy persists, which is
    the bug this function exists for; with it, the new value is used.
    """
    force = openmm.CustomExternalForce("k + 0*x")
    force.addGlobalParameter("k", 100.0)
    force.addParticle(0, [])
    omm.add_openmm_force(force)

    before = _total_energy()
    assert before == pytest.approx(100.0 / _KJ_PER_KCAL, rel=_REL_TOL)

    force.setGlobalParameterDefaultValue(0, 300.0)
    assert _total_energy() == pytest.approx(before, rel=_REL_TOL), (
        "CHARMM should not have noticed an edit made through the caller's "
        "own reference; if it did, system_changed() is redundant"
    )

    omm.system_changed()
    after = _total_energy()
    assert after - before == pytest.approx(
        (300.0 - 100.0) / _KJ_PER_KCAL, rel=_REL_TOL), (
        "system_changed() did not make CHARMM rebuild with the edited force"
    )


def test_bad_force_index_does_not_abort_the_run():
    """A mistyped force index must be reported, not fatal.

    The store used to index with std::vector::at behind a try/catch. The
    catch never ran -- two C++ runtimes in one process, so the exception
    unwound past it -- and a bad index ended the whole run with
    "terminating due to uncaught exception of type std::out_of_range".
    Reachable from a script by nothing worse than a typo.
    """
    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    index = omm.add_openmm_force(force)

    # "was it on before?" is False for a force that does not exist, which is
    # what these returned before -- when they returned at all.
    assert omm.force_turn_off(999) is False
    assert omm.force_turn_on(999) is False
    assert omm.force_turn_off(-1) is False

    # Still running, and the real force untouched by the bad calls.
    assert _total_energy() == pytest.approx(
        100.0 / _KJ_PER_KCAL, rel=_REL_TOL), (
        "a bad index disturbed the force at a valid index"
    )
    assert index == 0


def test_system_changed_is_harmless_when_nothing_changed():
    """Calling it needlessly costs a rebuild, not a wrong answer."""
    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    omm.add_openmm_force(force)

    before = _total_energy()
    omm.system_changed()
    assert _total_energy() == pytest.approx(before, rel=_REL_TOL)
