"""Tests for adding a pre-built OpenMM force to CHARMM by object handoff.

Covers `omm.add_openmm_force`, the `omm.EtermBucket` energy-term override,
and the safeguards that protect the pointer handoff. A force built with the
openmm Python package must produce the same energy as one built through
pyCHARMM's own wrapper classes, and every misuse must raise a clear Python
error rather than crashing the run.
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import energy

openmm = pytest.importorskip("openmm")

# add_openmm_force refuses to hand a pointer between mismatched OpenMM builds,
# so on a mismatch every test here would fail in a way that looks like a real
# regression.  Skip instead, naming both versions.
_PY_OMM = openmm.__version__
_PY_VER = int(_PY_OMM.split(".")[0]) * 10 + int(_PY_OMM.split(".")[1])
if omm.omm_version() != _PY_VER:
    _c_major, _c_minor = divmod(omm.omm_version(), 10)
    pytest.skip(
        f"openmm package is {_PY_OMM} but CHARMM was built against OpenMM "
        f"{_c_major}.{_c_minor}; forces cannot be passed between different "
        f"OpenMM builds",
        allow_module_level=True,
    )

pytestmark = pytest.mark.requires_feature("OPENMM")

# Energy of "fx*x*x" with fx=50 kJ/mol/nm^2 for one atom at x=1 Angstrom,
# i.e. 50 * 0.1^2 kJ/mol converted to kcal/mol.
_EXPECTED_KCAL = 50.0 * 0.1 * 0.1 / 4.184

# These tests check which energy term a force lands in, not OpenMM's
# arithmetic, so the tolerance only has to be far tighter than a wrong or
# missing contribution.  It must stay loose enough for OpenMM's default
# single-precision platforms, where summing several forces differs from the
# exact value in the 8th significant figure.
_REL_TOL = 1e-6


@pytest.fixture(autouse=True)
def single_atom_at_x1():
    """One atom of a dummy type, displaced 1 Angstrom along +x.

    Clears OpenMM on the way out as well as on the way in, so a force
    registered by the last test in this module cannot leak into the energies
    of another module.
    """
    setup_single_atom_system()
    from pycharmm import coor

    pos = coor.get_positions()
    pos.iloc[0] = [1.0, 0.0, 0.0]
    coor.set_positions(pos)
    yield
    omm.clear()


def _external_force(scale=50.0):
    """A CustomExternalForce with energy ``fx*x*x`` on atom 0."""
    force = openmm.CustomExternalForce("fx*x*x")
    force.addPerParticleParameter("fx")
    force.addParticle(0, [scale])
    return force


def _term(name):
    """Energy of CHARMM term `name`, or 0.0 if the term is not active."""
    if name in energy.get_term_names():
        return energy.get_term_by_name(name)
    return 0.0


def test_added_force_contributes_expected_energy():
    """A force built in Python lands in its default term with the right value.

    This is the core guarantee: handing CHARMM the finished OpenMM object
    reproduces the energy the force is defined to have.
    """
    index = omm.add_openmm_force(_external_force())
    assert index >= 0

    energy.get_energy(omm=True)
    assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_eterm_override_moves_the_energy():
    """`eterm` reports the energy in the chosen term, not the class default."""
    index = omm.add_openmm_force(_external_force(), eterm=omm.EtermBucket.CFCV)
    assert omm.get_force_eterm(index) == omm.EtermBucket.CFCV

    energy.get_energy(omm=True)
    assert _term("CFCV") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)
    assert _term("CFEX") == pytest.approx(0.0, abs=1e-12)


def test_override_can_be_cleared():
    """Passing None restores the default term for the force's class."""
    index = omm.add_openmm_force(_external_force(), eterm=omm.EtermBucket.CFCV)
    omm.set_force_eterm(index, None)
    assert omm.get_force_eterm(index) is None

    energy.get_energy(omm=True)
    assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_two_forces_in_one_term_sum():
    """Forces sharing a term have their energies added together."""
    omm.add_openmm_force(_external_force(scale=10.0))
    omm.add_openmm_force(_external_force(scale=20.0))

    energy.get_energy(omm=True)
    expected = (10.0 + 20.0) * 0.1 * 0.1 / 4.184
    assert _term("CFEX") == pytest.approx(expected, rel=_REL_TOL)


def test_subclass_of_supported_force_is_accepted():
    """A user subclass of a supported force type is recognised, not refused."""

    class Pull(openmm.CustomExternalForce):
        pass

    force = Pull("fx*x*x")
    force.addPerParticleParameter("fx")
    force.addParticle(0, [50.0])

    index = omm.add_openmm_force(force)
    energy.get_energy(omm=True)
    assert index >= 0
    assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


# --- safeguards: each of these must raise, never crash the process ---


def test_rejects_object_that_is_not_a_force():
    """Anything that is not an openmm.Force is refused before the handoff.

    The C layer cannot make this check itself, so a miss here would mean
    handing CHARMM a pointer to an unrelated object.
    """
    with pytest.raises(TypeError, match="openmm.Force"):
        omm.add_openmm_force("not a force")


def test_rejects_unsupported_force_class():
    """A real force of a class pyCHARMM cannot add yet is refused by name."""
    with pytest.raises(TypeError, match="HarmonicBondForce"):
        omm.add_openmm_force(openmm.HarmonicBondForce())


def test_rejects_adding_the_same_force_twice():
    """Re-adding a force is refused instead of double-counting its energy."""
    force = _external_force()
    omm.add_openmm_force(force)
    with pytest.raises(RuntimeError, match="already been handed"):
        omm.add_openmm_force(force)


def test_rejects_force_owned_by_an_openmm_system():
    """A force an OpenMM System already owns is refused, not freed twice."""
    force = _external_force()
    system = openmm.System()
    system.addParticle(10.0)
    system.addForce(force)  # System takes ownership
    with pytest.raises(RuntimeError, match="already been handed"):
        omm.add_openmm_force(force)


@pytest.mark.parametrize("bad_eterm", [99, -3, 1000])
def test_rejects_out_of_range_eterm(bad_eterm):
    """An energy-term code outside EtermBucket is refused with the valid set."""
    with pytest.raises(ValueError, match="not a valid CHARMM energy term"):
        omm.add_openmm_force(_external_force(), eterm=bad_eterm)


def test_rejects_non_integer_eterm():
    """A non-numeric energy term is refused with a usable hint."""
    with pytest.raises(TypeError, match="EtermBucket"):
        omm.add_openmm_force(_external_force(), eterm="CFEX")


def test_bad_eterm_does_not_register_the_force():
    """A refused energy term leaves no force behind on its default term.

    The term is validated before the force is registered, so a rejected call
    must not quietly add the force anyway.
    """
    with pytest.raises(ValueError):
        omm.add_openmm_force(_external_force(), eterm=99)

    energy.get_energy(omm=True)
    assert _term("CFEX") == pytest.approx(0.0, abs=1e-12)


def test_unknown_force_index_raises():
    """Reading the term of a force that does not exist is an IndexError.

    CHARMM reports "no such force" distinctly from "no override", so this
    must not read back as "using the default term".
    """
    with pytest.raises(IndexError):
        omm.get_force_eterm(999)


def test_set_eterm_on_unknown_index_raises():
    """Setting a term on a force that does not exist is reported, not ignored."""
    with pytest.raises(RuntimeError, match="no stored force"):
        omm.set_force_eterm(999, omm.EtermBucket.CFEX)


def test_force_added_after_an_energy_still_counts():
    """A force added after an energy run is included, not silently dropped.

    CHARMM assigns energy terms while building its OpenMM system, and reuses
    an already-built system unless told otherwise. Adding a force has to
    invalidate that system, or the force is quietly ignored for the rest of
    the run -- with a plausible-looking energy and no error.
    """
    energy.get_energy(omm=True)          # build the system with no extra force
    baseline = _term("CFEX")
    assert baseline == pytest.approx(0.0, abs=1e-12)

    omm.add_openmm_force(_external_force())
    energy.get_energy(omm=True)
    assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_eterm_change_after_an_energy_takes_effect():
    """Changing a force's term after an energy run moves the energy.

    Same hazard as adding a force late: the term is chosen during system
    build, so the change must force a rebuild rather than being ignored.
    """
    index = omm.add_openmm_force(_external_force())
    energy.get_energy(omm=True)
    assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)

    omm.set_force_eterm(index, omm.EtermBucket.CFCV)
    energy.get_energy(omm=True)
    assert _term("CFCV") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)
    assert _term("CFEX") == pytest.approx(0.0, abs=1e-12)


def test_rejected_add_leaves_python_owning_the_force():
    """A refused add must not disown the force, or it would leak.

    Ownership is handed over only after CHARMM accepts the force, so that a
    rejected object is still freed normally by Python.
    """
    force = _external_force()
    assert force.thisown
    with pytest.raises(ValueError):
        omm.add_openmm_force(force, eterm=99)
    assert force.thisown, "a rejected force must stay owned by Python"


def test_custom_force_wrapper_set_and_get_eterm():
    """`CustomForce.set_eterm`/`get_eterm` work on pyCHARMM's own wrappers.

    The energy-term override is not limited to forces handed over from the
    openmm package; pyCHARMM's wrapper classes expose it too.
    """
    force = omm.CustomExternalForce("fx*x*x")
    force.add_per_particle_parameter("fx")
    force.add_particle(0, [50.0])

    assert force.get_eterm() is None
    force.set_eterm(omm.EtermBucket.CFCV)
    assert force.get_eterm() == omm.EtermBucket.CFCV

    energy.get_energy(omm=True)
    assert _term("CFCV") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


@pytest.mark.parametrize(
    "class_name",
    [
        "CustomAngleForce",
        "CustomBondForce",
        "CustomCentroidBondForce",
        "CustomCompoundBondForce",
        "CustomExternalForce",
        "CustomGBForce",
        "CustomHbondForce",
        "CustomManyParticleForce",
        "CustomNonbondedForce",
        "CustomTorsionForce",
        "CustomCVForce",
        "RMSDForce",
    ],
)
def test_supported_class_names_exist_in_openmm(class_name):
    """Every class pyCHARMM claims to support is a real openmm class.

    Guards against a typo or a renamed class leaving an entry that can never
    match, which would show up only as a puzzling rejection for a user.
    """
    assert class_name in omm._OPENMM_CLASS_TO_KIND
    assert hasattr(openmm, class_name)


def test_every_supported_class_is_a_force_subclass():
    """Each supported entry names an openmm.Force subclass.

    The C layer's type check assumes it is being handed a Force, so a
    non-Force entry in the table would defeat the Python-side guard.
    """
    for class_name in omm._OPENMM_CLASS_TO_KIND:
        cls = getattr(openmm, class_name, None)
        if cls is None:
            continue           # newer/older OpenMM may not have them all
        assert issubclass(cls, openmm.Force), f"{class_name} is not a Force"


def test_repeated_clear_and_readd_is_stable():
    """Adding forces, clearing, and adding again keeps giving right answers.

    Each cycle destroys the force store and builds a fresh OpenMM system, so
    this covers the add/clear/re-add path users hit in a script that runs
    several systems. It does not attempt to detect heap corruption; see the
    ownership invariant in ``source/openmm/forcesStore.h`` for why the store
    deliberately does not free its forces.
    """
    for _ in range(5):
        setup_single_atom_system()
        from pycharmm import coor

        pos = coor.get_positions()
        pos.iloc[0] = [1.0, 0.0, 0.0]
        coor.set_positions(pos)

        omm.add_openmm_force(_external_force())
        energy.get_energy(omm=True)
        assert _term("CFEX") == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)
        omm.clear()


def test_indices_restart_after_clear():
    """Store indices start over once OpenMM is cleared, as documented."""
    first = omm.add_openmm_force(_external_force())
    omm.clear()
    setup_single_atom_system()
    second = omm.add_openmm_force(_external_force())
    assert first == second == 0


def test_cv_force_variables_survive_the_copy():
    """A CustomCVForce's collective variables survive CHARMM's copy of it.

    CHARMM duplicates a stored CustomCVForce by copying its collective
    variables one concrete type at a time. A variable whose type was missing
    from that list used to be dropped silently, leaving the CV force's energy
    expression referring to a variable that no longer existed -- which OpenMM
    either rejects at context creation or evaluates to something wrong.

    The variable used here is a CustomExternalForce of ``x``, so with the atom
    at 1 Angstrom (0.1 nm) the CV is 0.1 and the CV force's energy is
    ``100*cv`` = 10 kJ/mol. Getting that value back proves the variable came
    through the copy intact.

    Only variable types that can be validly configured on this one-atom
    system are exercised; types needing donors, acceptors or per-particle
    parameters for every atom (Hbond, ManyParticle, GB) cannot be built here,
    and OpenMM rejects a malformed force before the copy path is reached.
    """
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])

    cv_force = openmm.CustomCVForce("100*cv")
    cv_force.addCollectiveVariable("cv", cv)
    omm.add_openmm_force(cv_force)

    energy.get_energy(omm=True)
    expected = 100.0 * 0.1 / 4.184          # 100*cv kJ/mol -> kcal/mol
    assert _term("CFCV") == pytest.approx(expected, rel=_REL_TOL)


def test_eterm_bucket_values_match_charmm_abi():
    """EtermBucket codes are an ABI contract with fstore.F90's FB_* values."""
    assert omm.EtermBucket.CFIN == 0
    assert omm.EtermBucket.CFNB == 1
    assert omm.EtermBucket.CFEX == 2
    assert omm.EtermBucket.CFMB == 3
    assert omm.EtermBucket.CFCV == 4
    assert omm.EtermBucket.NNPO == 5
