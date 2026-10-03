"""A TorchScript potential driven through CHARMM by an ``openmm.PythonForce``.

``test_openmm_external_context.py`` already proves a Python-written force is
evaluated at all, on one atom with a hand-written quadratic. This file covers
what users actually do with that machinery, which is where two of them have got
stuck: a real molecule, a TorchScript model differentiated by autograd, and
positions and energies that have to cross a unit boundary on the way in and out.

What it adds beyond the existing test:

  * a 22-atom molecule rather than a single particle, so a mistake in the
    particle ordering or count has somewhere to show;
  * the energy checked against the model's own value on the same coordinates,
    rather than a number written down here -- so this stays correct on any
    platform, in any precision, and if the stand-in potential is ever changed;
  * the forces checked against finite differences, which is the only one of
    these that catches a sign flip or a missing nm-to-angstrom factor;
  * the fact that the ML energy lands in the VDW term, which is true, surprising
    and documented, so it is pinned here rather than rediscovered.

Nothing in this file is a captured reference value.

Run with:
    CHARMM_LIB_DIR=.../lib python -m pytest test_openmm_python_force_torch.py -v
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pycharmm.omm as omm

pytestmark = pytest.mark.requires_feature("OPENMM")

openmm = pytest.importorskip("openmm")

# Ask for a piece of torch that a directory holding only libraries cannot
# provide. A bare `importorskip("torch")` passes on such a tree and the failure
# surfaces later, during collection, taking the whole run with it.
pytest.importorskip("torch.nn")
pytest.importorskip("torch")

_PY_OMM = openmm.__version__
_PY_VER = int(_PY_OMM.split(".")[0]) * 10 + int(_PY_OMM.split(".")[1])
if omm.omm_version() != _PY_VER:
    _c_major, _c_minor = divmod(omm.omm_version(), 10)
    pytest.skip(
        f"openmm package is {_PY_OMM} but CHARMM was built against OpenMM "
        f"{_c_major}.{_c_minor}",
        allow_module_level=True,
    )

# A units or sign mistake is off by a factor -- a sign flip by 200%, a missing
# nm-to-angstrom conversion by 900% -- never by parts per million, so these can
# be loose enough to survive any precision model a platform might use.
REL_TOL = 1.0e-3
FD_REL_TOL = 1.0e-2

# Finite differences need a deliberately large step. The model is evaluated in
# float32, so the difference of two nearly equal energies loses precision as the
# step shrinks; measured on this stand-in, whose central differences carry no
# truncation error at all because it is quadratic:
#
#     step (nm)   1e-5      1e-4      1e-3      1e-2
#     rel error   1.4e-3    3.0e-4    2.5e-5    1.8e-6
#
# That is roundoff, not a wrong force. 1e-3 nm sits well below the noise while
# leaving room for the truncation error a real, non-quadratic potential adds.
FD_STEP_NM = 1.0e-3


# It has to be a subprocess for the same reason the PythonForce test in
# test_openmm_external_context.py is: when this machinery fails it does so as a
# segmentation fault, which would take pytest down with it.
_SNIPPET = textwrap.dedent('''
    import os
    import sys
    sys.path.insert(0, sys.argv[1])
    os.chdir(sys.argv[1])

    import numpy as np
    import torch
    import openmm
    from openmm import unit

    import pycharmm.omm as omm
    from pycharmm import NonBondedScript, coor, energy, gen, ic, psf, read

    FD_STEP_NM = float(sys.argv[2])
    MODEL_PATH = sys.argv[3]

    from _ml_pyforce_helpers import (
        ANGSTROM_PER_NM, EV_TO_KJ_PER_MOL as EV_TO_KJ, write_scripted_model)

    CALLS = {"n": 0}

    # Saved and reloaded rather than used directly, so the test goes through
    # torch.jit.load -- the path a user's own checkpoint takes.
    write_scripted_model(MODEL_PATH)
    model = torch.jit.load(MODEL_PATH)


    def model_energy_ev(coords_angstrom):
        """The model's own value, for the reference the test compares against."""
        with torch.no_grad():
            t = torch.tensor(coords_angstrom, dtype=torch.float32)
            return float(model(t, torch.zeros(len(t), dtype=torch.long)).item())


    def make_callback(natom):
        numbers = torch.zeros(natom, dtype=torch.long)

        def compute(state):
            CALLS["n"] += 1
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer)
            coords = torch.tensor(np.asarray(pos_nm) * ANGSTROM_PER_NM,
                                  dtype=torch.float32, requires_grad=True)
            e = model(coords, numbers).sum()
            (grad,) = torch.autograd.grad(e, coords)
            return (float(e.item()) * EV_TO_KJ,
                    -grad.detach().numpy().astype(np.float64)
                    * EV_TO_KJ * ANGSTROM_PER_NM)

        return compute


    # --- the molecule, as conftest builds it ---------------------------------
    read.rtf("data/top_all36_prot.rtf")
    read.prm("data/par_all36_prot.prm", flex=True)
    read.sequence_string("ALA")
    gen.new_segment("ADP", "ACE", "CT3", setup_ic=True)
    ic.prm_fill(False)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()
    coor.orient(by_rms=False, by_mass=False, by_noro=False)
    NonBondedScript(cutnb=18.0, ctonnb=15.0, ctofnb=13.0, eps=1.0, cdie=True,
                    atom=True, vatom=True, fswitch=True, vfswitch=True).run()

    natom = psf.get_natom()
    print("NATOM", natom)

    # Reference platform, so this runs anywhere and does not need a GPU.
    omm.set_platform("reference")
    omm.enable()

    system = openmm.System()
    omm.use_external_system(system)
    energy.show()
    print("PARTICLES", system.getNumParticles())
    print("MM_ONLY", repr(float(energy.get_total())))

    positions_a = np.asarray(coor.get_positions(), dtype=float)
    print("MODEL_KCAL", repr(float(model_energy_ev(positions_a) * EV_TO_KJ / 4.184)))

    system.addForce(openmm.PythonForce(make_callback(natom)))
    context = openmm.Context(
        system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
        openmm.Platform.getPlatformByName("Reference"))
    omm.use_external_context(context)

    CALLS["n"] = 0
    energy.show()
    print("WITH_MODEL", repr(float(energy.get_total())))
    print("CALLS", CALLS["n"])

    # --- forces, against finite differences, with no CHARMM in the way -------
    n_fd = 5
    rng = np.random.default_rng(20260923)
    pos_fd = rng.normal(scale=0.1, size=(n_fd, 3))
    fd_system = openmm.System()
    for _ in range(n_fd):
        fd_system.addParticle(1.0)
    fd_system.addForce(openmm.PythonForce(make_callback(n_fd)))
    fd_ctx = openmm.Context(
        fd_system, openmm.VerletIntegrator(1.0 * unit.femtosecond),
        openmm.Platform.getPlatformByName("Reference"))
    fd_ctx.setPositions(pos_fd * unit.nanometer)
    forces = fd_ctx.getState(getForces=True).getForces(asNumpy=True
        ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    worst = 0.0
    for i in range(n_fd):
        for d in range(3):
            shifted_energy = {}
            for sign in (+1, -1):
                shifted = pos_fd.copy()
                shifted[i, d] += sign * FD_STEP_NM
                fd_ctx.setPositions(shifted * unit.nanometer)
                shifted_energy[sign] = fd_ctx.getState(
                    getEnergy=True).getPotentialEnergy().value_in_unit(
                        unit.kilojoule_per_mole)
            numerical = -(shifted_energy[+1]
                          - shifted_energy[-1]) / (2 * FD_STEP_NM)
            worst = max(worst, abs(numerical - forces[i, d]))
    print("FD_REL", repr(float(worst / max(float(np.abs(forces).max()), 1e-12))))
    print("REACHED_THE_END")
''')


@pytest.fixture(scope="module")
def ml_run(tmp_path_factory):
    """Run the whole sequence once; every test below reads its output."""
    model_path = tmp_path_factory.mktemp("mlforce") / "stand_in.jpt"
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET,
         str(Path(__file__).resolve().parent), str(FD_STEP_NM),
         str(model_path)],
        capture_output=True, text=True, timeout=900, check=False,
    )
    out = proc.stdout

    # Everything is printed through float() in the snippet: numpy 2 reprs a
    # float64 as "np.float64(1.6e-05)", which does not parse back.
    def value(name, cast=float):
        found = re.search(rf"^{name} (\S+)$", out, re.MULTILINE)
        return cast(found.group(1)) if found else None

    return {
        "proc": proc,
        "out": out,
        "natom": value("NATOM", int),
        "particles": value("PARTICLES", int),
        "mm_only": value("MM_ONLY"),
        "with_model": value("WITH_MODEL"),
        "model_kcal": value("MODEL_KCAL"),
        # First column of each "ENER EXTERN>" line is VDWaals. Taken from the
        # printed table rather than a parameter lookup: CETERM(VDW) is 'VDW '
        # with a trailing space, resolved by subenr rather than through the
        # parameter store, so `?VDW` is not available here.
        "vdw": [float(m) for m in re.findall(
            r"^ENER EXTERN>\s+(-?[\d.]+)", out, re.MULTILINE)],
        "calls": value("CALLS", int),
        "fd_rel": value("FD_REL"),
    }


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_the_run_completes(ml_run):
    """Nothing segfaulted and every stage was reached."""
    assert ml_run["proc"].returncode == 0, (
        f"exit {ml_run['proc'].returncode}:\n"
        f"{(ml_run['out'] + ml_run['proc'].stderr)[-3000:]}")
    assert "REACHED_THE_END" in ml_run["out"], \
        "the run stopped part way through"


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_charmm_filled_the_supplied_system(ml_run):
    """One particle per PSF atom, and no more.

    Pre-adding particles to the System is the other mistake users make here;
    it leaves twice as many as the molecule has.
    """
    assert ml_run["particles"] == ml_run["natom"], (
        f"System holds {ml_run['particles']} particles, PSF has "
        f"{ml_run['natom']}")


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_the_model_was_actually_evaluated(ml_run):
    assert ml_run["calls"], "the Python function was never called"


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_contribution_is_what_the_model_is_worth(ml_run):
    """The energy that reached CHARMM is the model's own, converted.

    Compared against the model evaluated on the same coordinates rather than a
    number recorded here, so this holds on any platform and in any precision.
    """
    contribution = ml_run["with_model"] - ml_run["mm_only"]
    assert contribution == pytest.approx(ml_run["model_kcal"], rel=REL_TOL), (
        f"the model is worth {ml_run['model_kcal']} kcal/mol on this geometry, "
        f"but CHARMM's total moved by {contribution}")


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_forces_agree_with_finite_differences(ml_run):
    """The check that catches a sign flip or a missing length conversion.

    An energy comparison cannot: a force that is negated, or scaled by ten,
    leaves the energy exactly right.
    """
    assert ml_run["fd_rel"] < FD_REL_TOL, (
        f"worst force component is off by {ml_run['fd_rel']:.2e} relative to "
        f"the largest force")


@pytest.mark.skipif(not hasattr(openmm, "PythonForce"),
                    reason="openmm.PythonForce needs OpenMM 8.5 or newer")
def test_the_energy_lands_in_the_vdw_term(ml_run):
    """Surprising, true, and therefore pinned.

    A PythonForce keeps OpenMM's default force group 0, which CHARMM maps to
    VDW (omm_ecomponents.F90), so the whole ML energy is reported on the VDW
    line. The total is right; the breakdown is not. If this ever stops being
    true the documentation that warns about it needs changing, so the test
    should fail rather than quietly improve.
    """
    assert len(ml_run["vdw"]) >= 2, (
        f"expected two energy tables, found {len(ml_run['vdw'])}")
    moved = ml_run["vdw"][1] - ml_run["vdw"][0]
    contribution = ml_run["with_model"] - ml_run["mm_only"]
    assert moved == pytest.approx(contribution, rel=REL_TOL), (
        f"the VDW term moved by {moved} but the total moved by "
        f"{contribution}; force group 0 may no longer map to VDW")
