"""CHARMM filling in an OpenMM System the caller created.

The first half of letting a script create the OpenMM Context itself. A Context
must be built on the System CHARMM actually evaluates, and CHARMM cannot hand
a System outward after the fact -- a raw ``OpenMM::System`` pointer cannot be
turned back into a working Python object. So the caller creates the System,
keeps it, and CHARMM fills that one in.

Two things have to be true for this to be worth anything: what CHARMM puts in
has to be visible in the caller's own object, and the energy has to come out
exactly as it does when CHARMM makes its own System. Both are checked here.

The rest of the file is about ownership, which is where this can go wrong
quietly: CHARMM must not destroy a System it did not create, and must not fill
one in twice, since ``import_psf`` adds every particle and ``fstore_setup``
adds every stored force without checking what is already there.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_external_system.py -v
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import energy, reset

pytestmark = pytest.mark.requires_feature("OPENMM")

openmm = pytest.importorskip("openmm")

_PY_OMM = openmm.__version__
_PY_VER = int(_PY_OMM.split(".")[0]) * 10 + int(_PY_OMM.split(".")[1])
if omm.omm_version() != _PY_VER:
    _c_major, _c_minor = divmod(omm.omm_version(), 10)
    pytest.skip(
        f"openmm package is {_PY_OMM} but CHARMM was built against OpenMM "
        f"{_c_major}.{_c_minor}; a System cannot be shared between different "
        f"OpenMM builds",
        allow_module_level=True,
    )

# A constant 100 kJ/mol external force on the single atom, in kcal/mol.
_EXPECTED_KCAL = 100.0 / 4.184
_REL_TOL = 1e-6


@pytest.fixture(autouse=True)
def _fresh():
    """One-atom system, OpenMM on, and no external arrangement left behind."""
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    yield
    reset.openmm()


def _add_force_and_evaluate():
    """Add a constant external force, evaluate, and return the total energy."""
    force = omm.CustomExternalForce("100.0 + 0*x")
    force.add_particle(0, [])
    energy.show()
    return energy.get_total()


def test_charmm_fills_in_the_callers_system():
    """The caller's own object must show what CHARMM put into it.

    If the particles and forces did not appear here, CHARMM would be filling
    in something else and a Context built on this object would evaluate the
    wrong thing -- which is the whole failure this feature exists to avoid.
    """
    mine = openmm.System()
    assert mine.getNumParticles() == 0
    assert mine.getNumForces() == 0

    omm.use_external_system(mine)
    _add_force_and_evaluate()

    assert mine.getNumParticles() == 1, "CHARMM did not import the PSF into it"
    assert mine.getNumForces() > 0, "CHARMM added no forces to it"
    kinds = [type(mine.getForce(i)).__name__
             for i in range(mine.getNumForces())]
    assert "CustomExternalForce" in kinds, kinds


def test_energy_matches_charmms_own_system():
    """Whose System it is must make no difference to the answer."""
    own = _add_force_and_evaluate()

    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    omm.use_external_system(openmm.System())
    supplied = _add_force_and_evaluate()

    assert supplied == own, (
        f"supplying the System changed the energy: {supplied!r} vs {own!r}")
    assert own == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_state_reports_supplied_then_built():
    """The state has to distinguish "not yet built" from "already built"."""
    assert omm._external_system_state() == 0

    omm.use_external_system(openmm.System())
    assert omm._external_system_state() == 1

    _add_force_and_evaluate()
    assert omm._external_system_state() == 2


def test_supplying_a_second_system_after_building_is_refused():
    """Refused rather than filled in twice.

    CHARMM cannot empty a System it does not own, and OpenMM offers no way to,
    so the alternative to refusing would be counting the whole structure
    twice.
    """
    omm.use_external_system(openmm.System())
    _add_force_and_evaluate()

    with pytest.raises(RuntimeError, match="already been filled in"):
        omm.use_external_system(openmm.System())


def test_re_evaluating_does_not_double_the_supplied_system():
    """Evaluating twice must not add the structure again.

    This is the failure the refusal exists to prevent, checked from the
    outside: the energy and the force count both have to stand still.
    """
    mine = openmm.System()
    omm.use_external_system(mine)
    first = _add_force_and_evaluate()
    forces_after_first = mine.getNumForces()
    particles_after_first = mine.getNumParticles()

    energy.show()
    again = energy.get_total()

    assert again == first, f"energy moved on re-evaluation: {again!r} vs {first!r}"
    assert mine.getNumForces() == forces_after_first, "forces were added twice"
    assert mine.getNumParticles() == particles_after_first, \
        "particles were added twice"


def test_the_supplied_system_survives_charmm_tearing_down():
    """CHARMM must not destroy a System it did not create.

    If it did, this object would be a freed pointer and touching it would
    crash the interpreter rather than fail a test.
    """
    mine = openmm.System()
    omm.use_external_system(mine)
    _add_force_and_evaluate()

    omm.clear()               # tears down CHARMM's context and system

    assert mine.getNumParticles() == 1, "the supplied System was damaged"
    assert mine.getNumForces() > 0
    mine.addParticle(1.0)     # still a live, usable object
    assert mine.getNumParticles() == 2


def test_clear_lets_go_so_a_fresh_system_works():
    """After clearing, supplying another System must start cleanly."""
    omm.use_external_system(openmm.System())
    _add_force_and_evaluate()

    omm.clear()
    assert omm._external_system_state() == 0

    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    second = openmm.System()
    omm.use_external_system(second)
    assert _add_force_and_evaluate() == pytest.approx(_EXPECTED_KCAL,
                                                     rel=_REL_TOL)
    assert second.getNumParticles() == 1


def test_releasing_returns_charmm_to_its_own_system():
    """Giving it back must leave CHARMM working as it did before."""
    omm.use_external_system(openmm.System())
    assert omm._external_system_state() == 1

    omm.release_external_system()
    assert omm._external_system_state() == 0

    assert _add_force_and_evaluate() == pytest.approx(_EXPECTED_KCAL,
                                                     rel=_REL_TOL)


def test_a_force_added_after_the_build_is_applied_without_a_rebuild():
    """Adding a force after the build works, and costs one addForce.

    This is the workflow that works on the normal path -- evaluate, look, add
    a restraint, evaluate -- so people will expect it here. CHARMM cannot
    rebuild a System it does not own, so it adds the new force to the System as
    it stands and reinitializes the Context, preserving state.

    Two things are checked, because either alone would pass for the wrong
    reason: the energy has to pick the force up, and exactly one force may be
    added -- re-running the whole setup would give the right energy only after
    doubling everything already there.
    """
    mine = openmm.System()
    omm.use_external_system(mine)
    before = _add_force_and_evaluate()
    forces_before = mine.getNumForces()

    extra = openmm.CustomExternalForce("50.0 + 0*x")
    extra.addParticle(0, [])
    omm.add_openmm_force(extra)

    energy.show()
    after = energy.get_total()

    assert after == pytest.approx(before + 50.0 / 4.184, rel=_REL_TOL), (
        f"the added force was not picked up: {before!r} -> {after!r}")
    assert mine.getNumForces() == forces_before + 1, (
        "the whole setup was re-run instead of the one new force being added")


def test_the_delta_leaves_the_coordinates_alone():
    """Applying a force delta must not disturb the run's state.

    The Context is reinitialized with preserveState for exactly this reason.
    """
    from pycharmm import coor

    omm.use_external_system(openmm.System())
    _add_force_and_evaluate()
    before = coor.get_positions().to_numpy().copy()

    extra = openmm.CustomExternalForce("50.0 + 0*x")
    extra.addParticle(0, [])
    omm.add_openmm_force(extra)
    energy.show()

    after = coor.get_positions().to_numpy()
    assert (before == after).all(), "applying the delta moved the atoms"


@pytest.mark.parametrize("attempt,match", [
    pytest.param(lambda: omm.set_force_eterm(0, omm.EtermBucket.CFCV),
                 "energy term", id="set-eterm"),
    pytest.param(lambda: omm.force_turn_off(0), "Disabling", id="turn-off"),
    pytest.param(lambda: omm.system_changed(), "rebuild", id="system-changed"),
])
def test_changes_that_cannot_be_applied_are_still_refused(attempt, match):
    """What is refused is narrower now, but it is still refused, not fatal.

    Adding a force can be applied to the System as it stands. Changing a force
    already copied into it cannot -- CHARMM would have to rebuild, which it
    cannot do to a System it does not own. Those must raise where the mistake
    is made rather than ending the run at the next evaluation.
    """
    omm.use_external_system(openmm.System())
    _add_force_and_evaluate()

    with pytest.raises(RuntimeError, match=match):
        attempt()


def test_a_caller_who_keeps_no_reference_does_not_crash():
    """The obvious spelling must not be a use-after-free.

    ``omm.use_external_system(openmm.System())`` leaves the caller holding
    nothing, so unless pyCHARMM keeps a reference the System is collected
    while CHARMM is still using it -- CHARMM stores only the pointer and
    cannot keep a Python object alive. This segfaulted the interpreter before
    that reference existed, which is how it was found: two tests in this file
    were written that way.
    """
    import gc

    omm.use_external_system(openmm.System())
    gc.collect()          # collect now rather than at some later moment

    assert _add_force_and_evaluate() == pytest.approx(_EXPECTED_KCAL,
                                                     rel=_REL_TOL)


def test_only_an_openmm_system_is_accepted():
    """A wrong type must be a Python error, not a bad pointer.

    CHARMM cannot tell a System from any other address it is handed, so this
    has to be stopped here or it becomes a crash later.
    """
    with pytest.raises(TypeError, match="needs an openmm.System"):
        omm.use_external_system("not a system")

    with pytest.raises(TypeError, match="needs an openmm.System"):
        omm.use_external_system(openmm.CustomExternalForce("x^2"))


def test_releasing_while_built_leaves_the_system_alive():
    """Releasing a supplied System must not let CHARMM destroy it.

    ``release_external_system`` clears the "supplied" flag, and teardown used
    to decide ownership by reading that flag afterwards -- so it saw a System
    it believed was its own and freed one the caller still held. The caller was
    left holding freed memory: reading the particle count back gave garbage
    (-1642 in one run) and the process then crashed.

    Both energies here are incidental; the check is that the System is still
    the object it was, before and after.
    """
    system = openmm.System()
    omm.use_external_system(system)
    energy.get_energy(omm=True)
    before = system.getNumParticles()
    assert before > 0

    omm.release_external_system()
    energy.get_energy(omm=True)          # CHARMM tears down and rebuilds here

    assert system.getNumParticles() == before, (
        "the supplied System was freed or corrupted by CHARMM's teardown"
    )
    # Still a working object, not just an intact-looking count.
    assert system.getNumForces() >= 0


def test_releasing_a_system_twice_is_harmless():
    """A second release has nothing to give back and must not disturb CHARMM."""
    system = openmm.System()
    omm.use_external_system(system)
    energy.get_energy(omm=True)
    n = system.getNumParticles()

    omm.release_external_system()
    omm.release_external_system()
    energy.get_energy(omm=True)

    assert system.getNumParticles() == n
    assert omm._external_system_state() == 0   # back to CHARMM's own


def test_releasing_a_context_that_was_never_supplied_is_harmless():
    """Releasing with nothing supplied must leave a working setup behind.

    The release path forgets CHARMM's own Context and marks the setup gone,
    which strands the Context and System CHARMM built. That leak is not
    something this can see -- what it checks is the visible half: energies
    still come out right afterwards, so the no-op call cannot break a run.
    """
    energy.get_energy(omm=True)          # CHARMM builds its own
    first = energy.get_total()

    omm.release_external_context()       # nothing was supplied

    energy.get_energy(omm=True)
    assert energy.get_total() == pytest.approx(first, rel=1e-10)
