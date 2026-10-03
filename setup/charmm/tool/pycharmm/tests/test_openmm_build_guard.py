"""Tests that pyCHARMM refuses OpenMM work politely on a no-OpenMM build.

CHARMM can be built without OpenMM, and in that build every ``api_omm``
entry point is a Fortran stub.  Those stubs do not return a status: they call
CHARMM's fatal-error path, which at the default bomb level ends the process.
So a wrapper that reaches one has already taken the run down -- the caller
cannot catch it, retry, or even print a message.

The Python layer is therefore the only place a no-OpenMM build can be
reported survivably, which is why every entry point checks the build before
calling into CHARMM.  These tests hold that line by making the build *look*
like it has no OpenMM (``omm_version()`` returns 0, exactly as a real
no-OpenMM build reports) and checking each entry point raises instead of
calling through.

Patching is what makes this testable at all: the real thing cannot be tested
in-process, because the failure it guards against would kill the test runner.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_build_guard.py -v
"""

import pytest

import pycharmm.omm as omm


@pytest.fixture
def no_openmm_build(monkeypatch):
    """Make this build report itself as having no OpenMM support.

    A CHARMM built without OpenMM returns 0 from ``omm_version()``; a failed
    build-time probe leaves 60, which is a different case and deliberately
    not covered here.
    """
    monkeypatch.setattr(omm, "omm_version", lambda: 0)


def test_custom_force_refuses_without_openmm(no_openmm_build):
    """A wrapper force must raise, not take the process down."""
    with pytest.raises(NotImplementedError, match="without OpenMM support"):
        omm.CustomExternalForce("k*x^2")


@pytest.mark.parametrize("factory", [
    pytest.param(lambda: omm.CustomBondForce("r^2"), id="bond"),
    pytest.param(lambda: omm.CustomAngleForce("theta^2"), id="angle"),
    pytest.param(lambda: omm.CustomTorsionForce("theta^2"), id="torsion"),
    pytest.param(lambda: omm.CustomCVForce("cv"), id="cv"),
    pytest.param(lambda: omm.CustomCompoundBondForce(2, "distance(p1,p2)"),
                 id="compound"),
    pytest.param(lambda: omm.CustomGBForce(), id="gb"),
])
def test_every_wrapper_kind_refuses_without_openmm(no_openmm_build, factory):
    """The guard sits in the shared base, so every kind inherits it."""
    with pytest.raises(NotImplementedError, match="without OpenMM support"):
        factory()


def test_rmsd_force_refuses_without_openmm(no_openmm_build):
    """RMSDForce has its own constructor, so it needs its own guard."""
    with pytest.raises(NotImplementedError, match="without OpenMM support"):
        omm.RMSDForce([[0.0, 0.0, 0.0]])


def test_rmsd_force_checks_the_build_before_its_arguments(no_openmm_build):
    """The build check must come first, or the error names the wrong problem.

    Reference positions of the wrong shape raise ValueError. On a build with
    no OpenMM that is the less useful of the two complaints, and fixing it
    would only get the caller as far as the process-ending stub.
    """
    with pytest.raises(NotImplementedError):
        omm.RMSDForce([0.0, 0.0, 0.0])          # wrong shape as well


def test_rg_force_reports_missing_openmm_not_an_old_version(no_openmm_build):
    """RGForce is version-gated, and version 0 must not read as "too old".

    ``requires_omm_version`` would otherwise say this CHARMM "was built
    against OpenMM 0.0" and advise upgrading the openmm package -- pointing
    at a version that was never the problem.
    """
    with pytest.raises(NotImplementedError, match="without OpenMM support"):
        omm.RGForce()


def test_version_gate_still_reports_a_genuinely_old_openmm(monkeypatch):
    """The build check must not swallow the case the gate exists for."""
    monkeypatch.setattr(omm, "omm_version", lambda: 83)
    with pytest.raises(NotImplementedError, match="requires OpenMM 8.4"):
        omm.requires_omm_version(84, "RGForce")


def test_guard_is_inert_on_a_real_openmm_build():
    """On this build the guard must not fire; the tests above prove nothing
    if it refuses everywhere."""
    if omm.omm_version() <= 0:
        pytest.skip("this CHARMM really has no OpenMM support")
    omm.requires_omm_version(0, "nothing")     # must not raise
