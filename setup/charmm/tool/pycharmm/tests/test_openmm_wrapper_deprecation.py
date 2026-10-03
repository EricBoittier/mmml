"""The wrapper force classes warn and name their replacement.

pyCHARMM's Custom*Force classes exist because forces used to be built on
CHARMM's C++ side, one hand-written entry point per parameter per force type.
A force built with the ``openmm`` package and handed over with
``add_openmm_force`` is the same C++ force at the same speed, with the whole
OpenMM API available instead of the part the wrappers reproduced. So the
wrappers are being retired.

A deprecation warning is only useful if it tells you what to write instead,
so these tests check the replacement shown is right for each class -- the
constructors do not all take the same arguments, and a suggestion the reader
has to correct is worse than none.

Note pytest.ini ignores DeprecationWarning globally; ``pytest.warns``
overrides that within its block, which is why these tests see them at all.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_wrapper_deprecation.py -v
"""

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm.omm as omm
from pycharmm import reset

pytestmark = pytest.mark.requires_feature("OPENMM")


@pytest.fixture(autouse=True)
def _fresh_system():
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    yield
    reset.openmm()


# Every wrapper class, with the arguments its own constructor needs.
_WRAPPERS = [
    pytest.param("CustomBondForce", lambda: omm.CustomBondForce("r^2"),
                 id="bond"),
    pytest.param("CustomAngleForce", lambda: omm.CustomAngleForce("theta^2"),
                 id="angle"),
    pytest.param("CustomTorsionForce",
                 lambda: omm.CustomTorsionForce("theta^2"), id="torsion"),
    pytest.param("CustomExternalForce",
                 lambda: omm.CustomExternalForce("x^2"), id="external"),
    pytest.param("CustomNonbondedForce",
                 lambda: omm.CustomNonbondedForce("r"), id="nonbonded"),
    pytest.param("CustomHbondForce",
                 lambda: omm.CustomHbondForce("distance(a1,d1)"), id="hbond"),
    pytest.param("CustomCVForce", lambda: omm.CustomCVForce("cv"), id="cv"),
    pytest.param("CustomCompoundBondForce",
                 lambda: omm.CustomCompoundBondForce(2, "distance(p1,p2)"),
                 id="compound"),
    pytest.param("CustomCentroidBondForce",
                 lambda: omm.CustomCentroidBondForce(2, "distance(g1,g2)"),
                 id="centroid"),
    pytest.param("CustomManyParticleForce",
                 lambda: omm.CustomManyParticleForce(3, "1"),
                 id="manyparticle"),
    pytest.param("CustomGBForce", lambda: omm.CustomGBForce(), id="gb"),
    pytest.param("RMSDForce", lambda: omm.RMSDForce([[0.0, 0.0, 0.0]]),
                 id="rmsd"),
]


@pytest.mark.parametrize("class_name,make", _WRAPPERS)
def test_wrapper_warns_and_names_the_replacement(class_name, make):
    """Each class warns, and points at add_openmm_force by name."""
    with pytest.warns(DeprecationWarning) as caught:
        make()

    messages = [str(w.message) for w in caught]
    assert any(f"omm.{class_name} is deprecated" in m for m in messages), (
        f"no deprecation notice naming {class_name}: {messages}")
    assert any("add_openmm_force" in m for m in messages), (
        "the warning does not say what to use instead")


@pytest.mark.parametrize("class_name,make", _WRAPPERS)
def test_suggested_replacement_uses_the_same_class_name(class_name, make):
    """The suggestion must be a real OpenMM class, not a made-up one.

    Every wrapper deliberately shares its name with the OpenMM class it
    stands in for, which is what makes the replacement a substitution rather
    than a translation.
    """
    openmm = pytest.importorskip("openmm")
    with pytest.warns(DeprecationWarning) as caught:
        make()

    message = "\n".join(str(w.message) for w in caught)
    assert f"openmm.{class_name}(" in message, message
    assert hasattr(openmm, class_name), (
        f"the warning suggests openmm.{class_name}, which does not exist")


def test_counted_constructors_show_the_count_first():
    """A compound-bond force takes a particle count before the expression.

    Showing only an energy expression here would be a suggestion that fails
    with a TypeError the moment anyone tried it.
    """
    with pytest.warns(DeprecationWarning) as caught:
        omm.CustomCompoundBondForce(2, "distance(p1,p2)")

    message = "\n".join(str(w.message) for w in caught)
    assert 'openmm.CustomCompoundBondForce(<particles per bond>, ' in message, \
        message


def test_gb_force_shows_no_constructor_arguments():
    """openmm.CustomGBForce() takes nothing; its energy comes from a method.

    A placeholder expression here would suggest a call that does not exist.
    """
    with pytest.warns(DeprecationWarning) as caught:
        omm.CustomGBForce()

    message = "\n".join(str(w.message) for w in caught)
    assert "openmm.CustomGBForce()" in message, message


@pytest.mark.parametrize("make", [
    pytest.param(lambda: omm.CustomBondForce("r^2"), id="via-base-init"),
    pytest.param(lambda: omm.RMSDForce([[0.0, 0.0, 0.0]]), id="own-init"),
])
def test_warning_is_attributed_to_the_caller_not_to_omm_py(make):
    """The warning must name the caller's line, not a line inside pyCHARMM.

    This is not cosmetic. Python shows a DeprecationWarning by default only
    when it appears to come from ``__main__``, so one attributed to omm.py is
    dropped silently in an ordinary script -- invisible to the very people it
    is written for. It also ruins the per-call-site dedup: every use of a
    class would share one "location" and warn at most once per session.

    The two classes here reach the warning through chains of different depth,
    which is why the stack level is counted rather than hard-coded.
    """
    with pytest.warns(DeprecationWarning) as caught:
        make()

    origins = [w.filename for w in caught]
    assert all(not f.endswith("omm.py") for f in origins), (
        f"warning blamed on pyCHARMM instead of the caller: {origins}")
    assert any(f.endswith(__file__.split("/")[-1]) for f in origins), (
        f"warning should point at this test file: {origins}")


def test_abstract_base_is_rejected_without_a_deprecation_notice():
    """CustomForce itself must raise, and must not recommend openmm.CustomForce.

    There is no such OpenMM class, so a notice here would name a replacement
    that cannot be followed -- on a path that is already an error.
    """
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with pytest.raises(TypeError, match="Cannot instantiate"):
            omm.CustomForce("r^2")

    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert not deprecations, [str(w.message) for w in deprecations]


def test_the_replacement_itself_does_not_warn():
    """add_openmm_force is what we are steering people to, so it stays quiet.

    A warning on the recommended path would leave no way to do the right
    thing quietly.
    """
    openmm = pytest.importorskip("openmm")
    py_ver = (int(openmm.__version__.split(".")[0]) * 10
              + int(openmm.__version__.split(".")[1]))
    if omm.omm_version() != py_ver:
        pytest.skip("openmm package does not match CHARMM's OpenMM")

    force = openmm.CustomExternalForce("x^2")
    force.addParticle(0, [])

    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        omm.add_openmm_force(force)

    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert not deprecations, [str(w.message) for w in deprecations]
