"""CHARMM driving an OpenMM Context the caller created.

The second half of the handoff. The caller creates the System, CHARMM fills it
in, and then the caller creates the Context on it -- which is what lets them
choose the platform, bring their own integrator, or run a force that has to be
driven from Python.

The hard part is not making it work once; it is proving which Context is being
driven. An energy that comes out right proves nothing on its own, because
CHARMM's own Context would also give the right answer -- that mistake is why
this file stamps a fingerprint into the caller's Context and checks CHARMM
overwrote it.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_external_context.py -v
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from _custom_forces_helpers import setup_single_atom_system

import pycharmm
import pycharmm.omm as omm
from pycharmm import coor, energy, reset

pytestmark = pytest.mark.requires_feature("OPENMM")

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

_PY_OMM = openmm.__version__
_PY_VER = int(_PY_OMM.split(".")[0]) * 10 + int(_PY_OMM.split(".")[1])
if omm.omm_version() != _PY_VER:
    _c_major, _c_minor = divmod(omm.omm_version(), 10)
    pytest.skip(
        f"openmm package is {_PY_OMM} but CHARMM was built against OpenMM "
        f"{_c_major}.{_c_minor}; objects cannot be shared between different "
        f"OpenMM builds",
        allow_module_level=True,
    )

_EXPECTED_KCAL = 100.0 / 4.184
_REL_TOL = 1e-6
_NM_PER_ANGSTROM = 0.1


@pytest.fixture
def filled_system():
    """A supplied System that CHARMM has filled in, ready for a Context.

    A Context has to be built on a System that already has its particles and
    forces, so every test here needs this sequence first.
    """
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()

    system = openmm.System()
    omm.use_external_system(system)
    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    omm.add_openmm_force(force)
    energy.show()                      # CHARMM fills the System in

    yield system
    reset.openmm()


def _context_on(system):
    """A Langevin Context on `system`, on the Reference platform."""
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    return openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference"))


def _x_from(context):
    """The first particle's x from `context`, in Angstrom."""
    pos = context.getState(getPositions=True).getPositions(asNumpy=True)
    return pos[0][0].value_in_unit(unit.nanometer) / _NM_PER_ANGSTROM


def test_charmm_drives_the_callers_context_not_its_own(filled_system):
    """The decisive test: whose Context is actually being evaluated?

    A correct energy proves nothing here -- CHARMM's own Context would give
    the same number. So the caller's Context is stamped with a position CHARMM
    could not have produced, and the test is whether CHARMM overwrote it.
    """
    context = _context_on(filled_system)
    context.setPositions([[9.0, 9.0, 9.0]])       # nowhere near the real atom
    omm.use_external_context(context)

    energy.show()

    assert _x_from(context) != pytest.approx(90.0, abs=1e-6), (
        "the fingerprint survived, so CHARMM evaluated some other Context")
    assert energy.get_total() == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_energy_is_unchanged_by_who_owns_the_context(filled_system):
    """Handing the Context over must not change the answer."""
    energy.show()
    before = energy.get_total()

    omm.use_external_context(_context_on(filled_system))
    energy.show()

    assert energy.get_total() == before


def test_stochastic_dynamics_runs_through_the_callers_context(filled_system):
    """Langevin dynamics, which is what most CHARMM scripts actually run.

    Checked by agreement rather than by "it did not crash": CHARMM's
    coordinates and the caller's Context must describe the same atom
    afterwards.
    """
    context = _context_on(filled_system)
    omm.use_external_context(context)
    start = coor.get_positions().to_numpy().copy()

    pycharmm.DynamicsScript(
        start=True, lang=True, nstep=20, timestep=0.002, iasors=1, iasvel=1,
        firstt=300.0, finalt=300.0, tbath=300.0, nprint=20, echeck=-1.0,
        omm=True,
    ).run()

    charmm_x = coor.get_positions().to_numpy()[0][0]
    assert charmm_x == pytest.approx(_x_from(context), abs=1e-4), (
        "CHARMM and the caller's Context disagree about where the atom is")
    assert abs(charmm_x - start[0][0]) > 1e-6, "the atom never moved"
    assert omm._external_context_state() == 1


def test_a_second_dynamics_command_works(filled_system):
    """Two commands in a row exercise the per-command clean start.

    CHARMM gives its own Context clean integrator state per command by
    rebuilding it. It cannot rebuild this one, so it reinitializes discarding
    state and pushes the run state straight back -- and if that push were
    missed, OpenMM would refuse to step with "Particle positions have not
    been set", which is exactly what happened while this was being written.
    """
    context = _context_on(filled_system)
    omm.use_external_context(context)

    for iasvel in (1, 0):
        pycharmm.DynamicsScript(
            start=True, lang=True, nstep=20, timestep=0.002, iasors=1,
            iasvel=iasvel, firstt=300.0, finalt=300.0, tbath=300.0,
            nprint=20, echeck=-1.0, omm=True,
        ).run()
        assert coor.get_positions().to_numpy()[0][0] == pytest.approx(
            _x_from(context), abs=1e-4)


def test_adopting_a_second_context_leaves_the_first_alive(filled_system):
    """Only a Context CHARMM built is CHARMM's to destroy.

    On a second adoption the Context being replaced is the caller's, so
    freeing it would hand them a dangling pointer -- a crash, not a failure.
    """
    first = _context_on(filled_system)
    second = _context_on(filled_system)

    omm.use_external_context(first)
    energy.show()
    omm.use_external_context(second)
    energy.show()

    assert first.getSystem().getNumParticles() == 1, \
        "the first Context was damaged"
    assert energy.get_total() == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_releasing_returns_charmm_to_its_own_context(filled_system):
    """Handing it back must leave CHARMM working as before."""
    omm.use_external_context(_context_on(filled_system))
    assert omm._external_context_state() == 1

    omm.release_external_context()
    assert omm._external_context_state() == 0

    energy.show()
    assert energy.get_total() == pytest.approx(_EXPECTED_KCAL, rel=_REL_TOL)


def test_a_context_on_the_wrong_system_is_refused(filled_system):
    """The quietest possible failure, refused loudly.

    A Context built on some other System has none of the forces CHARMM added,
    so its energies would be wrong rather than obviously broken.
    """
    stranger = openmm.System()
    stranger.addParticle(1.0)

    with pytest.raises(RuntimeError, match="different System"):
        omm.use_external_context(_context_on(stranger))


def test_a_context_is_refused_before_the_system_is_filled_in():
    """Order matters, and the error has to say so.

    A Context built on an empty System would have no particles and no
    forces.
    """
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    system = openmm.System()
    omm.use_external_system(system)          # CHARMM has not filled it in

    # A particle of the caller's own, only so that OpenMM will build a Context
    # at all -- it refuses on an empty System ("Cannot create a Context for a
    # System with no particles"), which is why the guard cannot be reached
    # with a genuinely empty one.
    system.addParticle(1.0)

    with pytest.raises(RuntimeError, match="not been filled in"):
        omm.use_external_context(_context_on(system))
    reset.openmm()


def test_a_context_is_refused_without_a_supplied_system():
    """Supplying only a Context is not enough, and says why."""
    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()

    system = openmm.System()
    system.addParticle(1.0)
    with pytest.raises(RuntimeError, match="no System has been supplied"):
        omm.use_external_context(_context_on(system))
    reset.openmm()


def test_only_an_openmm_context_is_accepted(filled_system):
    """A wrong type must be a Python error, not a bad pointer."""
    with pytest.raises(TypeError, match="needs an openmm.Context"):
        omm.use_external_context("not a context")

    with pytest.raises(TypeError, match="needs an openmm.Context"):
        omm.use_external_context(filled_system)


# Run in a subprocess because the refusal below ends the CHARMM run by design:
# the requested change cannot be applied, and continuing would compute
# something the caller did not ask for.  Same reason the BLaDE restart
# validation tests spawn rather than call.
_NBOPTS_SNIPPET = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import openmm
    from openmm import unit
    import pycharmm.omm as omm
    from pycharmm import energy, lingo
    from _custom_forces_helpers import setup_single_atom_system

    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()
    system = openmm.System()
    omm.use_external_system(system)
    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    omm.add_openmm_force(force)
    energy.show()

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    omm.use_external_context(openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")))
    energy.show()

    lingo.charmm_script("nbonds cutnb 10.0 ctofnb 9.0 ctonnb 8.0")
    energy.show()
    print("REACHED_THE_END")
""")


def test_changing_nonbonded_options_is_refused_with_a_named_reason(tmp_path):
    """Nonbonded options are built into the System, so they cannot be applied.

    Honouring them means rebuilding the System, which CHARMM cannot do to one
    it did not create. Before this was handled here, the attempt tore the
    Context down and hit omm_create_system's refusal instead -- ending the run
    while naming neither the nonbonded options nor anything the user could do
    about it. `nbonds` is ordinary in scripts, so the message has to be the
    useful one.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _NBOPTS_SNIPPET,
         str(Path(__file__).resolve().parent)],
        capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr

    assert "REACHED_THE_END" not in out, \
        "the nonbonded change was accepted, so the System is now wrong"
    assert "CHECK_NBOPTS" in out, out[-2000:]
    assert "cannot be rebuilt" in out, out[-2000:]
    assert "OMM_CREATE_SYSTEM" not in out, \
        "still failing with the unhelpful message instead of the named one"


# A force whose energy comes from a Python function is the reason the external
# System/Context machinery exists, so it is worth a test of its own -- and it
# has to be a subprocess, because the way this fails is a segmentation fault
# that would take pytest down with it.
_PYTHON_FORCE_SNIPPET = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    disable_guard = len(sys.argv) > 2 and sys.argv[2] == "no-guard"

    import numpy as np
    import openmm
    from openmm import unit
    import pycharmm.omm as omm
    from pycharmm import energy
    from _custom_forces_helpers import setup_single_atom_system

    if disable_guard:
        omm._ensure_openmm_numpy_c_api = lambda: None

    K = 50.0                      # kJ/mol/nm^2
    CALLS = {"n": 0}

    def compute(state):
        CALLS["n"] += 1
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        x = float(pos[0, 0])
        forces = np.zeros_like(pos)
        forces[0, 0] = -2.0 * K * x
        return K * x * x, forces

    setup_single_atom_system()
    # Off the origin: the helper leaves the atom at 0, where K*x^2 is zero and
    # a broken force is indistinguishable from a working one.
    from pycharmm import coor
    positions = coor.get_positions()
    positions.iloc[0] = [1.0, 0.0, 0.0]
    coor.set_positions(positions)

    omm.set_platform("reference")
    omm.enable()

    system = openmm.System()
    omm.use_external_system(system)
    energy.get_energy(omm=True)
    baseline = energy.get_total()

    force = openmm.PythonForce(compute)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference"))
    omm.use_external_context(context)

    CALLS["n"] = 0
    energy.get_energy(omm=True)
    print("CONTRIBUTION", repr(energy.get_total() - baseline))
    print("CALLS", CALLS["n"])
""")

_PYTHON_FORCE_X_NM = 0.1                    # the snippet moves the atom to 1 A
_PYTHON_FORCE_KCAL = 50.0 * _PYTHON_FORCE_X_NM ** 2 / 4.184


def _run_python_force(*extra):
    return subprocess.run(
        [sys.executable, "-c", _PYTHON_FORCE_SNIPPET,
         str(Path(__file__).resolve().parent), *extra],
        capture_output=True, text=True, timeout=300,
    )


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_charmm_can_evaluate_a_force_written_in_python():
    """The point of the whole external System/Context path.

    A force whose energy is computed by a Python function, evaluated because
    CHARMM asked for an energy. Checked by the number, not by "it did not
    crash": the contribution has to be K*x^2 for the atom where CHARMM put it.
    """
    proc = _run_python_force()

    assert proc.returncode == 0, (
        f"the run died (exit {proc.returncode}) instead of returning an "
        f"energy:\n{(proc.stdout + proc.stderr)[-3000:]}")
    out = proc.stdout
    assert "CALLS 0" not in out, "the Python function was never called"
    contribution = float(out.split("CONTRIBUTION")[1].split()[0])
    assert contribution == pytest.approx(_PYTHON_FORCE_KCAL, rel=1e-9), \
        "CHARMM's energy does not contain what the Python function returned"


# The bug this canary watches was fixed upstream by openmm/openmm#5400 and
# released in 8.6.1, so from there on the run is supposed to survive and there
# is nothing left to prove. Below it the bug is live and the canary must fire.
_NUMPY_C_API_FIXED = (8, 6, 1)


def _openmm_release():
    """(major, minor, patch) from openmm.__version__, missing parts as 0.

    Release strings carry suffixes -- 8.7.0beta, 8.6.1.dev0 -- so each field is
    read up to its first non-digit rather than passed straight to int().
    """
    fields = []
    for piece in openmm.__version__.split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        fields.append(int(digits) if digits else 0)
    return tuple(fields + [0] * (3 - len(fields)))


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
@pytest.mark.skipif(_openmm_release() >= _NUMPY_C_API_FIXED,
                    reason="fixed upstream in OpenMM 8.6.1 (openmm/openmm#5400)")
def test_the_python_force_guard_is_still_needed():
    """A canary on the OpenMM bug the guard works around.

    OpenMM asks whether the forces the callback returned are a numpy array,
    through a C API table that nothing on this path initialises, so without
    the guard the run dies with no message. If this test ever fails because
    the run *survived*, OpenMM has fixed it upstream and
    `_ensure_openmm_numpy_c_api` can be deleted -- which is the only way we
    would find that out.

    That is what happened: it fired on 2026-09-14 against a fresh OpenMM, and
    the fix is openmm/openmm#5400 in 8.6.1. The test now skips itself there and
    keeps watching the older releases that still have the bug, because the
    guard stays until pyCHARMM's oldest supported OpenMM is past it.
    """
    proc = _run_python_force("no-guard")

    # A negative returncode is death by signal (-11, SIGSEGV, is what this
    # bug does). Accepting any non-zero exit would let the test pass on an
    # ImportError or a renamed helper, which would prove nothing at all --
    # and this test exists only to prove one specific thing.
    assert proc.returncode < 0, (
        f"expected the run to be killed by a signal, got exit "
        f"{proc.returncode}. If it is 0, OpenMM has fixed this upstream and "
        f"_ensure_openmm_numpy_c_api can be removed. Anything else means the "
        f"run failed for an unrelated reason and this test proved "
        f"nothing:\n{(proc.stdout + proc.stderr)[-2000:]}")


def _child_env():
    """Environment pinning a subprocess to the library THIS process is using.

    Left to itself a child resolves its own pycharmm and its own libchmm --
    which in a shared environment need not be the ones under test, so the
    test can quietly examine a different build and report whatever that one
    does.  Point it at both explicitly instead.
    """
    import os

    import pycharmm.loader as _loader

    env = dict(os.environ)
    lib = Path(_loader._loader.charmm_lib_name)
    if lib.is_absolute():
        env["CHARMM_LIB_DIR"] = str(lib.parent)
    pkg_parent = str(Path(pycharmm.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


# A thermostat or barostat is a FORCE inside the System, added while CHARMM
# populates it -- which for a supplied System happens before any dynamics
# options exist.  So asking for one on a supplied Context cannot work, and it
# used to fail silently: the run went ahead with no thermostat at all, having
# just printed "Constant temperature w/ OpenMM using Andersen heatbath
# requested".  This has to be a subprocess because refusing is a wrndie, which
# takes the interpreter down with it.
_MISSING_THERMOSTAT_SNIPPET = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import openmm
    from openmm import unit
    from _custom_forces_helpers import setup_single_atom_system

    import pycharmm
    import pycharmm.omm as omm
    from pycharmm import energy

    setup_single_atom_system()
    omm.set_platform("reference")
    omm.enable()

    system = openmm.System()
    omm.use_external_system(system)
    force = openmm.CustomExternalForce("100.0 + 0*x")
    force.addParticle(0, [])
    omm.add_openmm_force(force)
    energy.show()

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference"))
    omm.use_external_context(context)

    import pycharmm.loader as _loader
    print("CHILD_LIB", _loader._loader.charmm_lib_name, flush=True)

    names = [type(system.getForce(i)).__name__
             for i in range(system.getNumForces())]
    print("FORCES_BEFORE", names, flush=True)

    pycharmm.DynamicsScript(
        start=True, nstep=10, timestep=0.001, iasors=1, iasvel=1,
        firstt=300.0, finalt=300.0, tbath=300.0, nprint=10, echeck=-1.0,
        omm=True, andersen=True, colfrq=10,
    ).run()

    names = [type(system.getForce(i)).__name__
             for i in range(system.getNumForces())]
    print("FORCES_AFTER", names, flush=True)
    print("REACHED_THE_END", flush=True)
""")


def test_andersen_on_a_supplied_context_is_refused_not_ignored():
    """Asking for a thermostat CHARMM cannot add must fail, not run silently.

    The thermostat is a force in the System, added when CHARMM populates it
    -- before any dynamics options exist, for a supplied System.  It cannot
    be added afterwards either, because the caller's Context is already
    built on that System.  So the only honest outcome is to refuse.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _MISSING_THERMOSTAT_SNIPPET,
         str(Path(__file__).resolve().parent)],
        capture_output=True, text=True, timeout=300,
        env=_child_env(),
    )
    out = proc.stdout + proc.stderr

    assert "FORCES_BEFORE" in out, out[-2000:]

    # A subprocess that resolved a different libchmm would be reporting on a
    # different build, so say that rather than let it look like a behaviour
    # difference.  This has actually happened: in a tree whose pycharmm came
    # from the working directory the child picked up another library.
    import pycharmm.loader as _loader
    parent_lib = _loader._loader.charmm_lib_name
    child_lib = next(
        (ln.split(None, 1)[1].strip()
         for ln in out.splitlines() if ln.startswith("CHILD_LIB")), None)
    assert child_lib == parent_lib, (
        f"the subprocess loaded {child_lib!r} but this process is testing "
        f"{parent_lib!r}, so its result says nothing about this build")

    assert "AndersenThermostat" not in out.split("FORCES_BEFORE")[1][:200], \
        "the supplied System already had a thermostat, so this proves nothing"
    assert "REACHED_THE_END" not in out, (
        "dynamics ran to completion with no thermostat force -- the Andersen "
        "request was silently dropped")
    assert "ANDErsen heatbath requested" in out, out[-2000:]
