"""Example: letting your script own the OpenMM System and Context.

WHAT THIS IS FOR

By default CHARMM builds the OpenMM System and Context itself and you never
see them. That is the right thing almost always -- it needs no setup, and
everything in this file is unnecessary.

Three situations are not covered by the default, and this is what they need:

1. **You want a specific platform, or platform properties.** ``omm.set_platform``
   picks a platform by name, but nothing lets you set platform *properties* --
   CUDA precision, a device index, a deterministic-forces flag. Those are
   arguments to the OpenMM Context constructor, so the Context has to be
   yours.

2. **You want an integrator CHARMM does not offer.** CHARMM builds five kinds
   from the ``DYNAmics`` options. If you want a Nose-Hoover chain, a custom
   integrator, or a Middle-scheme variant CHARMM does not select, you have to
   build it and hand it over inside a Context.

3. **You want a force that runs your own Python code.** OpenMM's PythonForce
   only works in a Context that Python created, so this is the prerequisite
   for driving a force from Python at all. This is the machine-learned
   potential case, and step 4 below does it. It needs OpenMM 8.5 or newer.

If none of those is what you are after, stop here and let CHARMM do it.

HOW IT WORKS, AND WHY IN THIS ORDER

A Context is built on a System, and the System has to have CHARMM's particles
and forces in it before the Context is made. CHARMM also cannot hand a System
outward after the fact -- a raw C++ System pointer cannot be turned back into
a usable Python object. So the order is:

    omm.enable()                      # route energies through OpenMM
    system = openmm.System()          # you create it, empty
    omm.use_external_system(system)   # CHARMM will fill in THIS one
    energy.show()                     # ...and does, here
    ctx = openmm.Context(system, integrator, platform, properties)
    omm.use_external_context(ctx)     # CHARMM drives it from now on

omm.enable() has to come first. Supplying the System does not by itself route
the energy through OpenMM, so without it the energy above is an ordinary CHARMM
energy, the System is never filled in, and use_external_context refuses it with
"the System you supplied has not been filled in yet". The platform the run uses
is the one on your Context, so it does not have to be named here.

WHO OWNS WHAT

    You own          the System object, the Context, the integrator inside it,
                     and the platform choice.
    CHARMM owns      what goes into the System (particles from the PSF,
                     restraints, stored forces, thermostat and barostat), and
                     the coordinates.

CHARMM pushes positions, velocities, time and box vectors in before every
evaluation, so anything you set there is overwritten. Set coordinates through
CHARMM (``coor.set_positions``), not through the Context.

Because the integrator is yours, the timestep, temperature and friction in use
are the ones you built it with -- not the ones on the ``DYNAmics`` command.
CHARMM says so when it notices the difference, and its own time accounting
still comes from the command, so keeping the two in step is your job.

WHAT IS REFUSED, AND WHY

    A Context built on a different System   -- it would have none of CHARMM's
                                               forces, so energies would be
                                               quietly wrong
    A Context before the System is filled   -- it would have no particles
    Changing a force already in the System  -- CHARMM's answer to that is to
                                               rebuild, which it cannot do to
                                               a System it does not own

Adding a force afterwards is fine: it is added to the System as it stands.

Run with:
    CHARMM_LIB_DIR=.../lib python example_openmm_external_context.py
"""

import sys

import numpy

import pycharmm
import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, energy, lingo, read
from pycharmm import generate as gen

try:
    import openmm
    from openmm import unit
except ImportError:
    sys.exit("This example needs the openmm Python package "
             "(conda install -c conda-forge openmm).")

# A single dummy atom, so the only interesting energy is the force we add.
_RTF = """
read rtf card
* single atom
*
   20    1
MASS     -1 X     10.0

RESI TEST       0.0
GROUP
ATOM A    X     0.0
PATC  FIRS NONE LAST NONE
END
"""

_PRM = """
read param card
* dummy parameters
*
NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
X        0.0440    1.0       0.8000

END
"""


def build_one_atom_system():
    """Build the minimal CHARMM structure this example runs on."""
    omm.clear()
    if psf.get_natom() > 0:
        lingo.charmm_script("delete atom sele all end")
    lingo.charmm_script(_RTF)
    lingo.charmm_script(_PRM)
    read.sequence_string("TEST")
    gen.new_segment(seg_name="MOL")
    positions = coor.get_positions()
    # Off the origin on purpose: the force below is -(fx*x), which is zero at
    # x = 0, and an example whose energies all read 0.000000 demonstrates
    # nothing.
    positions.iloc[0] = [1.0, 0.0, 0.0]
    coor.set_positions(positions)


def main():
    if omm.omm_version() <= 0:
        sys.exit("This CHARMM was built without OpenMM support.")

    build_one_atom_system()
    omm.set_platform("reference")
    omm.enable()

    # --- 1. create the System yourself, and let CHARMM fill it in ----------
    system = openmm.System()
    omm.use_external_system(system)
    print(f"before CHARMM fills it: {system.getNumParticles()} particles, "
          f"{system.getNumForces()} forces")

    pull = openmm.CustomExternalForce("-(fx*x)")
    pull.addPerParticleParameter("fx")
    pull.addParticle(0, [100.0])            # 100 kJ/mol/nm along +x
    omm.add_openmm_force(pull)

    energy.show()                            # CHARMM fills the System in here
    print(f"after  CHARMM fills it: {system.getNumParticles()} particles, "
          f"{system.getNumForces()} forces")
    print("forces CHARMM put in:",
          [type(system.getForce(i)).__name__
           for i in range(system.getNumForces())])

    # --- 2. build the Context, with the platform and integrator you want ---
    #
    # This is the part the default cannot do for you. Platform properties go
    # here; on CUDA you might pass
    #     {"Precision": "mixed", "DeviceIndex": "0"}
    platform = openmm.Platform.getPlatformByName("Reference")
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    context = openmm.Context(system, integrator, platform)

    omm.use_external_context(context)
    print(f"\nCHARMM is now driving your Context on the "
          f"{context.getPlatform().getName()} platform")

    energy.show()
    print(f"energy through your Context: {energy.get_total():.6f} kcal/mol")

    # --- 3. dynamics, using your integrator -------------------------------
    #
    # The timestep and temperature that matter are the integrator's, above.
    # CHARMM starts each command from clean integrator state, as it does for
    # its own Context.
    start = coor.get_positions().to_numpy().copy()
    pycharmm.DynamicsScript(
        start=True, lang=True, nstep=50, timestep=0.002, iasors=1, iasvel=1,
        firstt=300.0, finalt=300.0, tbath=300.0, nprint=50, echeck=-1.0,
        omm=True,
    ).run()
    moved = abs(coor.get_positions().to_numpy() - start).max()
    print(f"dynamics moved the atom {moved:.4f} A")

    # CHARMM's coordinates and your Context agree: it really is your Context.
    from_context = context.getState(getPositions=True).getPositions(
        asNumpy=True)[0][0].value_in_unit(unit.nanometer) * 10.0
    print(f"CHARMM says x = {coor.get_positions().to_numpy()[0][0]:.6f} A, "
          f"your Context says x = {from_context:.6f} A")

    # --- 4. a force whose energy is your own Python function ---------------
    #
    # The case the other three exist to make possible: CHARMM asks for an
    # energy, and your function is what answers.  Swap the arithmetic below
    # for a machine-learned model and nothing else changes.
    if hasattr(openmm, "PythonForce"):
        calls = {"n": 0}

        def my_energy(state):
            """E = K*x^2 on atom 0, in OpenMM's units (kJ/mol, nm)."""
            calls["n"] += 1
            pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            forces = numpy.zeros_like(pos)
            forces[0, 0] = -2.0 * 50.0 * pos[0, 0]
            return 50.0 * float(pos[0, 0]) ** 2, forces

        # Same sequence as above, from a clean start: a System can only be
        # filled in once, so the earlier one cannot be reused here.
        omm.release_external_context()
        build_one_atom_system()
        omm.set_platform("reference")
        omm.enable()
        system2 = openmm.System()
        omm.use_external_system(system2)
        energy.show()
        before = energy.get_total()

        system2.addForce(openmm.PythonForce(my_energy))
        context2 = openmm.Context(
            system2,
            openmm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                            1 / unit.picosecond,
                                            0.002 * unit.picoseconds),
            openmm.Platform.getPlatformByName("Reference"))
        omm.use_external_context(context2)

        calls["n"] = 0
        energy.show()
        print(f"\nyour Python function was called {calls['n']} time(s) "
              f"and contributed "
              f"{energy.get_total() - before:.6f} kcal/mol")
        print("(twice per energy: the second call fills the CHARMM energy "
              "table, which asks for each term separately)")
    else:
        print(f"\nopenmm {openmm.__version__} has no PythonForce "
              f"(needs 8.5 or newer), so skipping the Python-function force")

    # --- 5. handing control back ------------------------------------------
    #
    # Gives back the System as well: it cannot be filled in twice, so CHARMM
    # returns to making its own of both. Neither of your objects is destroyed.
    omm.release_external_context()
    energy.show()
    print(f"\nback on CHARMM's own Context: {energy.get_total():.6f} kcal/mol")
    print(f"your System is still yours: {system.getNumParticles()} particles")


if __name__ == "__main__":
    main()
