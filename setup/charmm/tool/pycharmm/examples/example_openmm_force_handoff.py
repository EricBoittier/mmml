"""Example: two ways to add a custom OpenMM force, and picking its energy term.

pyCHARMM can take a custom force either way:

1. Built with a pyCHARMM wrapper class (``omm.CustomExternalForce``). Stays in
   pyCHARMM style, needs no extra package, works on any CHARMM built with
   OpenMM. Prefer this unless you need the OpenMM API itself.

2. Built with the ``openmm`` Python package and handed over whole
   (``omm.add_openmm_force``). Prefer this when you are following OpenMM
   documentation, reusing OpenMM code, or need something the wrappers do not
   expose. It requires that the ``openmm`` package be the same OpenMM install
   CHARMM was built against, and refuses to run when it is not.

Both end up as the same C++ force inside OpenMM and run at the same speed.

The script also shows the energy-term choice: by default a force's energy is
reported in the term implied by its class (an external force lands in CFEX),
but ``eterm`` / ``set_eterm`` puts it in whichever term you name, so you can
watch one force on its own line.

Run with:
    CHARMM_LIB_DIR=.../lib python example_openmm_force_handoff.py
"""

import sys

import pycharmm.omm as omm
import pycharmm.psf as psf
from pycharmm import coor, energy, lingo, read
from pycharmm import generate as gen

# One atom of a dummy type, so the only interesting energy is the force we add.
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


def build_one_atom_at_x(x_angstrom=1.0):
    """Set up a fresh one-atom system with the atom displaced along +x.

    Called before each section so they do not interfere with each other.
    ``omm.clear()`` drops the OpenMM state and any forces added to it, but it
    does not touch the structure, so the atoms from the previous section have
    to be deleted explicitly -- otherwise they pile up and the later sections
    are no longer running on one atom.

    Parameters
    ----------
    x_angstrom : float, optional
        Where to put the atom on the x axis, in Angstrom. Default 1.0.
    """
    omm.clear()
    if psf.get_natom() > 0:
        lingo.charmm_script("delete atom sele all end")
    lingo.charmm_script(_RTF)
    lingo.charmm_script(_PRM)
    read.sequence_string("TEST")
    gen.new_segment(seg_name="MOL")
    pos = coor.get_positions()
    pos.iloc[0] = [x_angstrom, 0.0, 0.0]
    coor.set_positions(pos)


def show_term(label, name):
    """Print one CHARMM energy term, or note that it is not active.

    Parameters
    ----------
    label : str
        Text to print in front of the value.
    name : str
        Four-character CHARMM energy term name, e.g. ``"CFEX"``.

    Returns
    -------
    float
        The term's energy in kcal/mol, or 0.0 if CHARMM does not currently
        list that term name at all. A listed term reads 0.0 on its own when
        nothing contributes to it, which is what the CFEX line shows once the
        force has been moved to another term.
    """
    if name in energy.get_term_names():
        value = energy.get_term_by_name(name)
    else:
        value = 0.0
    print(f"    {label:<38} {name} = {value: .6f} kcal/mol")
    return value


def route_1_pycharmm_wrapper():
    """Add a restraint with pyCHARMM's own wrapper class.

    Returns
    -------
    float
        The CFEX energy the force produced, in kcal/mol.
    """
    print("\n1. pyCHARMM wrapper class (omm.CustomExternalForce)")
    build_one_atom_at_x(1.0)

    # k*x^2 in OpenMM units: k in kJ/mol/nm^2, x in nm.
    force = omm.CustomExternalForce("k*x*x")
    force.add_per_particle_parameter("k")
    force.add_particle(0, [50.0])

    energy.get_energy(omm=True)
    return show_term("energy in the default term", "CFEX")


def route_2_openmm_handoff():
    """Add the same restraint built with the openmm package.

    Returns
    -------
    float
        The CFEX energy the force produced, in kcal/mol.
    """
    print("\n2. openmm package, handed over (omm.add_openmm_force)")
    try:
        import openmm
    except ImportError:
        print("    openmm package not installed; skipping this route")
        return None

    build_one_atom_at_x(1.0)

    # Same force, written with OpenMM's own camelCase API.
    force = openmm.CustomExternalForce("k*x*x")
    force.addPerParticleParameter("k")
    force.addParticle(0, [50.0])

    try:
        index = omm.add_openmm_force(force)
    except RuntimeError as exc:
        # Most likely the openmm package and CHARMM's OpenMM differ.
        print(f"    could not hand the force over: {exc}")
        return None

    print(f"    force stored at index {index}")
    energy.get_energy(omm=True)
    return show_term("energy in the default term", "CFEX")


def choosing_the_energy_term():
    """Report the same force in a term of our choosing instead of the default.

    Returns
    -------
    tuple of float
        The (CFEX, CFCV) energies after the override.
    """
    print("\n3. Choosing which energy term the force reports in")
    build_one_atom_at_x(1.0)

    force = omm.CustomExternalForce("k*x*x")
    force.add_per_particle_parameter("k")
    force.add_particle(0, [50.0])

    # An external force would normally report in CFEX.  Put it in CFCV so it
    # can be followed separately from any other external forces.
    force.set_eterm(omm.EtermBucket.CFCV)
    print(f"    energy term pinned to {force.get_eterm().name}")

    energy.get_energy(omm=True)
    cfex = show_term("default term is now empty", "CFEX")
    cfcv = show_term("energy shows up here instead", "CFCV")
    return cfex, cfcv


def safeguards():
    """Show the errors raised for the common mistakes, instead of crashing."""
    print("\n4. Safeguards (each raises rather than crashing the run)")
    try:
        import openmm
    except ImportError:
        print("    openmm package not installed; skipping")
        return

    build_one_atom_at_x(1.0)

    # Not an OpenMM force at all.  RuntimeError is caught alongside TypeError
    # here and below because add_openmm_force checks that Python's OpenMM
    # matches CHARMM's first; on a mismatched build that check fires before
    # the one being demonstrated.
    try:
        omm.add_openmm_force("not a force")
    except (TypeError, RuntimeError) as exc:
        print(f"    not a force        -> {type(exc).__name__}: {str(exc)[:52]}...")

    # A force class pyCHARMM does not handle yet.
    try:
        omm.add_openmm_force(openmm.HarmonicBondForce())
    except (TypeError, RuntimeError) as exc:
        print(f"    unsupported class  -> {type(exc).__name__}: {str(exc)[:52]}...")

    # Handing the same force over twice would double-count its energy.
    force = openmm.CustomExternalForce("k*x*x")
    force.addPerParticleParameter("k")
    force.addParticle(0, [50.0])
    try:
        omm.add_openmm_force(force)
        omm.add_openmm_force(force)
    except RuntimeError as exc:
        print(f"    added twice        -> RuntimeError: {str(exc)[:54]}...")

    # An energy term that does not exist.
    try:
        omm.set_force_eterm(0, 99)
    except (ValueError, RuntimeError) as exc:
        print(f"    bad energy term    -> {type(exc).__name__}: {str(exc)[:52]}...")


def main():
    """Run every section and check the two routes agree.

    Returns
    -------
    int
        0 on success, 1 if the two routes disagreed.
    """
    print(__doc__.split("Run with:")[0].rstrip())

    wrapper_energy = route_1_pycharmm_wrapper()
    handoff_energy = route_2_openmm_handoff()
    choosing_the_energy_term()
    safeguards()

    print("\nSummary")
    if handoff_energy is None:
        print("    only the wrapper route ran; nothing to compare")
        return 0
    print(f"    wrapper class  CFEX = {wrapper_energy: .6f} kcal/mol")
    print(f"    openmm handoff CFEX = {handoff_energy: .6f} kcal/mol")
    if abs(wrapper_energy - handoff_energy) < 1e-6:
        print("    the two routes agree, as they should")
        return 0
    print("    MISMATCH: the two routes should give the same energy")
    return 1


if __name__ == "__main__":
    sys.exit(main())
