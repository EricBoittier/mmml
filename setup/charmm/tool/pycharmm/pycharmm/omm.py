# pycharmm: molecular dynamics in python with CHARMM
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""OpenMM integration for pyCHARMM.

Functions
=========
Control Functions:
- `enable` -- Enable OpenMM for GPU-accelerated calculations
- `disable` -- Disable OpenMM (keep context)
- `clear` -- Disable OpenMM and destroy context
- `is_enabled` -- Check if OpenMM is enabled
- `check_restraints` -- Check active restraints for OpenMM compatibility

Configuration Functions:
- `set_platform` -- Set OpenMM platform (cuda, opencl, cpu, reference)
- `set_device` -- Set GPU device ID
- `set_precision` -- Set precision model (single, mixed, double)
- `get_supported_restraints` -- Get restraint compatibility info

Pre-built OpenMM Force Functions:
- `add_openmm_force` -- Add a force built with the openmm Python package
- `set_force_eterm` -- Choose which CHARMM energy term a force reports in
- `get_force_eterm` -- Read back a force's energy-term override

Serialization Functions:
- `get_system_serial` -- Get serialized version of current charmm/openmm system

Torch Functions:
- `torch_add_force` -- Add a torch force to CHARMM's OpenMM system
- `torch_outputs_forces` -- Set torch force output mode
- `torch_add_global_param` -- Add global parameter to torch force
- `torch_set_global_param` -- Set global parameter value

Classes
=======
- `CustomBondForce` -- custom bond interactions
- `CustomAngleForce` -- custom angle interactions
- `CustomTorsionForce` -- custom torsion interactions
- `CustomExternalForce` -- custom external potential
- `CustomNonbondedForce` -- custom nonbonded interactions
- `CustomCompoundBondForce` -- custom multi-particle bond interactions
- `CustomCentroidBondForce` -- custom centroid-based interactions
- `CustomGBForce` -- custom generalized Born force
- `CustomHbondForce` -- custom hydrogen bond interactions
- `CustomManyParticleForce` -- custom many-particle interactions
- `CustomCVForce` -- custom collective variable force

Energy term buckets
===================
Each user-added custom force contributes to one of five CHARMM energy
term slots, grouped by the kind of OpenMM force.  This lets you see
custom-force contributions broken out in `printe` output and via
`pycharmm.energy.get_term_by_name()`, instead of all being lumped
together.  TorchForce continues to report into the dedicated `NNPO`
slot.  The mapping is exposed as `CUSTOM_FORCE_BUCKETS`:

- `CFIN` -- internal coords (BOND / ANGLE / TORSION custom forces)
- `CFNB` -- nonbonded (NONBONDED / GB)
- `CFEX` -- external (EXTERNAL)
- `CFMB` -- many-body (COMPOUND_BOND / CENTROID_BOND / H_BOND / MANY_PARTICLE)
- `CFCV` -- collective variables (CV / VOLUME / RMSD / RG)

Forces in the same bucket share one OpenMM force-group bit, so their
energies are summed by a single `getState` call per active bucket.

The energy term is chosen from the force's kind by default, but it does
not have to be.  `set_force_eterm` (or the `eterm` argument of
`add_openmm_force`) reports any force in the term you name, which is
useful when you want a force's energy on its own line rather than added
in with everything else of the same kind.

Two ways to add a custom force
==============================
There are two ways to give CHARMM an OpenMM custom force, and they are
interchangeable as far as the simulation is concerned -- both are
evaluated by OpenMM in C++ at the same speed.  Pick whichever is more convenient:

**Use pyCHARMM's own wrapper classes** (`omm.CustomExternalForce`,
`omm.CustomBondForce`, ...) when you want to stay in pyCHARMM style and
the force is one the wrappers cover.  They take CHARMM-flavoured
snake_case methods, need no extra package, and work in any CHARMM built
with OpenMM:

>>> f = omm.CustomExternalForce("k*x^2")        # doctest: +SKIP
>>> _ = f.add_per_particle_parameter("k")      # doctest: +SKIP
>>> _ = f.add_particle(0, [100.0])             # doctest: +SKIP

**Use `add_openmm_force`** when you want the OpenMM API itself -- because
you are following OpenMM documentation or an OpenMM example, reusing code
or a force object from another OpenMM project, or you need something the
wrappers do not expose (for instance a tabulated function on a force
class pyCHARMM has not wired up).  You build the force with the `openmm`
package and hand the finished object over:

>>> import openmm                              # doctest: +SKIP
>>> f = openmm.CustomExternalForce("k*x^2")    # doctest: +SKIP
>>> _ = f.addPerParticleParameter("k")         # doctest: +SKIP
>>> _ = f.addParticle(0, [100.0])              # doctest: +SKIP
>>> i = omm.add_openmm_force(f)                # doctest: +SKIP

`add_openmm_force` requires the `openmm` Python package and CHARMM to be
using the same OpenMM library, since a force object cannot cross between
two different builds.  As a guard it compares the two OpenMM versions and
refuses on a mismatch, which catches the common case of an environment
whose `openmm` was upgraded after CHARMM was built.  Matching versions do
not prove the libraries are the same file, so if a run crashes inside
OpenMM just after a force is added, check which `libOpenMM` CHARMM was
linked against.  If you are unsure, or writing an example for other people
to run, prefer the wrapper classes.

Whichever route you use, add forces before the energy or dynamics command
that should feel them, build a fresh force object for each add, and add
them again after `clear()`.

Examples
========
>>> import pycharmm
>>> import pycharmm.omm as omm

Get serlialized system
>>> my_sys_xml = omm.get_system_serial()

Inspect the bucket a custom-force kind lands in:
>>> omm.CUSTOM_FORCE_BUCKETS[omm.CustomForceType.BOND]
'CFIN'

Read a bucket energy after a force evaluation:
>>> from pycharmm import energy
>>> energy.get_term_by_name('CFIN')   # all custom BOND/ANGLE/TORSION forces
"""

import ctypes
import enum
import sys
import warnings

import pycharmm.restraints as restraints
from pycharmm.lingo import charmm_script
from pycharmm.loader import lib

# Unit conversion constants (CHARMM uses Angstrom/kcal; OpenMM uses nm/kJ)
NM_PER_ANGSTROM = 0.1
ANGSTROM_PER_NM = 10.0
KJ_PER_KCAL = 4.184
KCAL_PER_KJ = 1.0 / 4.184

# Module-level state
_omm_enabled = False
_omm_config = {}


class OmmRestraintWarning(UserWarning):
    """Warning issued when active restraints may be incompatible with OpenMM."""
    pass


class OmmEngineError(Exception):
    """Error related to OpenMM engine operations."""
    pass


def check_restraints():
    """Check if active restraints are compatible with OpenMM.

    Returns
    -------
    list[str]
        List of incompatible restraint type names, empty if all compatible.
    """
    from pycharmm.restraints import _state
    return _state._check_backend_conflicts('openmm')


def enable(platform=None, deviceid=None, precision=None, nocpu=False,
           warn_restraints=True, raise_on_incompatible=False):
    """Enable OpenMM for GPU-accelerated energy and dynamics calculations.

    Parameters
    ----------
    platform : str, optional
        OpenMM platform: 'cuda', 'opencl', 'cpu', 'reference'.
    deviceid : int or str, optional
        GPU device ID(s). Can be single ID (0) or multiple ('0,1').
    precision : str, optional
        Precision model: 'single', 'mixed', 'double'.
    nocpu : bool, optional
        If True, fail if non-GPU platform is selected. Default False.
    warn_restraints : bool, optional
        If True (default), warn if incompatible restraints are active.
    raise_on_incompatible : bool, optional
        If True, raise OmmEngineError for incompatible restraints.

    Raises
    ------
    OmmEngineError
        If raise_on_incompatible=True and incompatible restraints are active.
    """
    global _omm_enabled, _omm_config

    conflicts = check_restraints()
    if conflicts:
        msg = (
            f"Active restraints incompatible with OpenMM: {', '.join(conflicts)}. "
            f"These restraints will not be evaluated correctly on GPU."
        )
        if raise_on_incompatible:
            raise OmmEngineError(msg)
        elif warn_restraints:
            warnings.warn(msg, OmmRestraintWarning, stacklevel=2)

    try:
        restraints.set_backend('openmm')
    except restraints.IncompatibleBackendError:
        if raise_on_incompatible:
            raise

    charmm_script("OMM ON")
    _omm_enabled = True

    if platform is not None:
        set_platform(platform)
    if deviceid is not None:
        set_device(deviceid)
    if precision is not None:
        set_precision(precision)
    if nocpu:
        charmm_script("OMM NOCPU")

    _omm_config = {
        'platform': platform,
        'deviceid': deviceid,
        'precision': precision,
        'nocpu': nocpu
    }


def disable():
    """Disable OpenMM but retain the OpenMM Context.

    Use `clear()` to fully destroy the context.
    """
    global _omm_enabled
    charmm_script("OMM OFF")
    _omm_enabled = False
    restraints.set_backend('standard')


def clear():
    """Disable OpenMM and destroy the OpenMM Context.

    Notes
    -----
    ``OMM CLEAR`` also makes CHARMM let go of a System supplied through
    :func:`use_external_system`, so the reference pyCHARMM was holding on the
    caller's behalf is dropped here to match. The System itself is not
    destroyed -- it still belongs to whoever created it.
    """
    global _omm_enabled, _omm_config, _supplied_system_ref
    global _supplied_context_ref
    charmm_script("OMM CLEAR")
    _omm_enabled = False
    _omm_config = {}
    _supplied_system_ref = None
    _supplied_context_ref = None
    restraints.set_backend('standard')


def is_enabled():
    """Check if OpenMM is currently enabled.

    Returns
    -------
    bool
        True if OpenMM is enabled, False otherwise.
    """
    return _omm_enabled


def set_platform(platform):
    """Set the OpenMM platform.

    Parameters
    ----------
    platform : str
        Platform name: 'cuda', 'opencl', 'cpu', 'reference'.
    """
    valid_platforms = ('cuda', 'opencl', 'cpu', 'reference')
    if platform.lower() not in valid_platforms:
        raise ValueError(f"Platform must be one of {valid_platforms}, got '{platform}'")
    charmm_script(f"OMM PLATFORM {platform}")


def set_device(deviceid):
    """Set the GPU device(s) to use.

    Parameters
    ----------
    deviceid : int or str
        GPU device ID(s). Single int (0) or string ('0,1').
    """
    charmm_script(f"OMM DEVICEID {deviceid}")


def set_precision(precision):
    """Set the precision model for OpenMM calculations.

    Parameters
    ----------
    precision : str
        Precision model: 'single', 'mixed', 'double'.
    """
    valid_precisions = ('single', 'mixed', 'double')
    if precision.lower() not in valid_precisions:
        raise ValueError(f"Precision must be one of {valid_precisions}, got '{precision}'")
    charmm_script(f"OMM PRECISION {precision}")


def get_config():
    """Get current OpenMM configuration.

    Returns
    -------
    dict
        Dictionary with current OpenMM configuration.
    """
    return _omm_config.copy()


def get_supported_restraints():
    """Get information about restraint support in OpenMM.

    Returns
    -------
    dict
        Dictionary with 'supported' and 'unsupported' lists.
    """
    return {
        'supported': [
            'CONS HARM ABSO - Absolute harmonic restraints (XSCALE=YSCALE=ZSCALE=1)',
            'CONS RESD - Restrained distances',
            'CONS DIHE - Dihedral restraints'
        ],
        'unsupported': [
            'NOE/PNOE - Distance restraints not implemented',
            'IC - Internal coordinate restraints not implemented',
            'DROPLET - Droplet restraints not implemented',
            'CONS HARM (relative/bestfit) - Only absolute restraints supported'
        ],
        'notes': {
            'harmonic': 'Only ABSOLUTE restraints with XSCALE=YSCALE=ZSCALE=1 supported.',
            'precision': 'Single precision may have energy drift. Use mixed or double.',
            'periodic': 'Only orthorhombic boxes supported (alpha=beta=gamma=90).'
        }
    }


def get_system_serial():
    """Get the serialized version of current charmm/openmm system

       This returns a string of characters in a specific xml format.
       The string can then be read into a python OpenMM simmulation.

    Returns
    -------
    str
        A serialized representation of the current charmm/openmm system

    """
    max_size = lib.omm_get_system_serial_size()
    buffer = ctypes.create_string_buffer(max_size)
    lib.omm_get_system_serial(buffer, max_size)
    return buffer.value.decode(errors="ignore")



def force_turn_on(index):
    """Enable a stored force so it is used next time OpenMM is set up.

    Parameters
    ----------
    index : int
        Index of the force in CHARMM's force store.

    Returns
    -------
    bool
        True if the force was already enabled.  Also False for an index that
        does not exist, so the return value does not confirm the force was
        found.
    """
    _refuse_if_supplied_system_built("Enabling a stored force")
    index = ctypes.c_int(index)
    old_status = lib.api_force_turn_on(index)
    status = False
    if old_status == 1:
        status = True

    return status


def force_turn_off(index):
    """Disable a stored force so it is left out of the OpenMM system.

    The force stays in the store and can be switched back on with
    `force_turn_on`, so this is the way to drop a force's contribution
    without rebuilding it.

    Parameters
    ----------
    index : int
        Index of the force in CHARMM's force store.

    Returns
    -------
    bool
        True if the force was enabled before this call.  Also False for an
        index that does not exist, so the return value does not confirm the
        force was found.
    """
    _refuse_if_supplied_system_built("Disabling a stored force")
    index = ctypes.c_int(index)
    old_status = lib.api_force_turn_off(index)
    status = False
    if old_status == 1:
        status = True

    return status


class CustomForceType(enum.Enum):
    """OpenMM custom-force kinds CHARMM's force store can hold.

    Each member's value is the integer CHARMM uses for that kind across the C
    interface, so these numbers are an ABI contract with
    ``ForcesStore::ForceType`` in ``source/openmm/forcesStore.h``; use the
    members, not the numbers.  `CUSTOM_FORCE_BUCKETS` maps each kind to the
    CHARMM energy term it reports in by default.
    """

    TORCH = 0
    ANGLE = 1
    BOND = 2
    CV = 3
    CENTROID_BOND = 4
    COMPOUND_BOND = 5
    EXTERNAL = 6
    GB = 7
    H_BOND = 8
    MANY_PARTICLE = 9
    NONBONDED = 10
    TORSION = 11
    VOLUME = 12
    RMSD = 13
    RG = 14


# Maps each CustomForceType to the 4-character CHARMM CETERM name of the
# energy bucket its contribution lands in.  TorchForce keeps its dedicated
# NNPO slot; user-added OpenMM custom forces are summed into one of five
# CF* buckets — see source/openmm/fstore.F90 fstore_setup for the
# Fortran-side dispatch and source/energy/energym.F90 for the slot indices.
CUSTOM_FORCE_BUCKETS = {
    CustomForceType.TORCH:          "NNPO",
    CustomForceType.BOND:           "CFIN",
    CustomForceType.ANGLE:          "CFIN",
    CustomForceType.TORSION:        "CFIN",
    CustomForceType.NONBONDED:      "CFNB",
    CustomForceType.GB:             "CFNB",
    CustomForceType.EXTERNAL:       "CFEX",
    CustomForceType.COMPOUND_BOND:  "CFMB",
    CustomForceType.CENTROID_BOND:  "CFMB",
    CustomForceType.H_BOND:         "CFMB",
    CustomForceType.MANY_PARTICLE:  "CFMB",
    CustomForceType.CV:             "CFCV",
    CustomForceType.VOLUME:         "CFCV",
    CustomForceType.RMSD:           "CFCV",
    CustomForceType.RG:             "CFCV",
}


class EtermBucket(enum.IntEnum):
    """CHARMM energy terms an OpenMM force's energy can be reported in.

    Pass one of these to :func:`add_openmm_force` or :func:`set_force_eterm`
    to choose which CHARMM energy term a force contributes to, instead of
    the default implied by the force's class (see `CUSTOM_FORCE_BUCKETS`).

    **When to prefer naming a term.** Name one when you want a force's energy
    reported separately -- either to watch one force on its own, or to split
    several forces of the same kind across different terms. Otherwise leave
    the default: it already groups forces sensibly by kind. The terms are
    fixed slots in CHARMM's energy table, not free-form labels, so pick the
    one whose meaning is closest to your force; the energy is reported there
    regardless of what kind of force it actually is.

    Forces sharing a term have their energies summed into it. The values are
    the codes understood by CHARMM's ``api_cf_set_bucket``; they must stay in
    step with the ``FB_*`` parameters in ``source/openmm/fstore.F90``.

    Attributes
    ----------
    CFIN : int
        Internal-coordinate term, the default for bond/angle/torsion forces.
    CFNB : int
        Nonbonded term, the default for nonbonded and GB forces.
    CFEX : int
        External-potential term, the default for external forces.
    CFMB : int
        Many-body term, the default for compound/centroid/hbond/many-particle
        forces.
    CFCV : int
        Collective-variable term, the default for CV/volume/RMSD/Rg forces.
    NNPO : int
        Neural-network-potential term. Only available in CHARMM builds with
        OpenMM-Torch support; in other builds CHARMM refuses the override and
        keeps the force's default term.
    """

    CFIN = 0
    CFNB = 1
    CFEX = 2
    CFMB = 3
    CFCV = 4
    NNPO = 5


# Maps an OpenMM force class name to its CustomForceType. Used by
# add_openmm_force to work out the force kind on the user's behalf, so a
# caller cannot pair a force with a mismatched kind. Names (not classes) are
# used so this module never has to import openmm.
_OPENMM_CLASS_TO_KIND = {
    "CustomAngleForce":        CustomForceType.ANGLE,
    "CustomBondForce":         CustomForceType.BOND,
    "CustomCVForce":           CustomForceType.CV,
    "CustomCentroidBondForce": CustomForceType.CENTROID_BOND,
    "CustomCompoundBondForce": CustomForceType.COMPOUND_BOND,
    "CustomExternalForce":     CustomForceType.EXTERNAL,
    "CustomGBForce":           CustomForceType.GB,
    "CustomHbondForce":        CustomForceType.H_BOND,
    "CustomManyParticleForce": CustomForceType.MANY_PARTICLE,
    "CustomNonbondedForce":    CustomForceType.NONBONDED,
    "CustomTorsionForce":      CustomForceType.TORSION,
    "CustomVolumeForce":       CustomForceType.VOLUME,
    "RMSDForce":               CustomForceType.RMSD,
    "RGForce":                 CustomForceType.RG,
}

# OpenMM force classes that duplicate something CHARMM builds for itself,
# mapped to what they would duplicate. Adding one of these would put a second
# copy of that interaction into the system: the energy is then counted twice,
# at whatever settings the handed-over force happens to carry, with nothing to
# warn the user. These are refused permanently -- not "not supported yet".
#
# Classes NOT in this map are refused with the milder "pyCHARMM cannot add one
# of these yet", which is the honest answer for an OpenMM force that does not
# collide with CHARMM's own terms.
_CHARMM_OWNED_CLASSES = {
    "NonbondedForce":
        "the nonbonded interactions (electrostatics and van der Waals)",
    "HarmonicBondForce": "bond stretching",
    "HarmonicAngleForce": "angle bending",
    "PeriodicTorsionForce": "dihedral torsions",
    "RBTorsionForce": "dihedral torsions",
    "CMAPTorsionForce": "CMAP corrections",
    "GBSAOBCForce": "generalised-Born implicit solvent",
    "DrudeForce": "Drude polarisable interactions",
    "AndersenThermostat": "temperature control",
    "MonteCarloBarostat": "pressure control",
    "MonteCarloAnisotropicBarostat": "pressure control",
    "MonteCarloMembraneBarostat": "pressure control",
    "CMMotionRemover": "centre-of-mass motion removal",
}

# Where to set the same thing through CHARMM instead, for the cases where one
# command is the obvious answer. Absent means "set it wherever you normally
# would in CHARMM", which is all that can honestly be said.
_CHARMM_OWNED_REMEDY = {
    "NonbondedForce": "the NBONds command (cutoffs, dielectric, PME, and so on)",
    "GBSAOBCForce": "CHARMM's own implicit-solvent commands (GBSW, GBMV, ...)",
    "AndersenThermostat": "the thermostat options on the DYNAmics command",
    "MonteCarloBarostat": "the pressure options on the DYNAmics command",
    "MonteCarloAnisotropicBarostat":
        "the pressure options on the DYNAmics command",
    "MonteCarloMembraneBarostat":
        "the pressure options on the DYNAmics command",
}

# Status codes returned by api_custom_force_add_ptr / api_cf_set_bucket.
# Must match the FSTORE_ERR_* macros in source/openmm/forcesStore.h. CHARMM
# already prints an explanation for each; these give the Python exception a
# matching summary.
_FSTORE_ERRORS = {
    -1: "CHARMM's OpenMM force store does not exist yet -- enable OpenMM "
        "first (omm.enable()).",
    -2: "the force pointer was null -- the OpenMM force object was not "
        "created, or has already been freed.",
    -3: "CHARMM rejected the force type -- either it does not match the "
        "force's class, or this CHARMM was built without support for that "
        "force type.",
    -4: "the energy term code is out of range.",
    -5: "there is no stored force with that index.",
}


def _stacklevel_of_caller():
    """Return the ``stacklevel`` that points at the first frame outside here.

    A warning is only useful if it names the line the reader has to change,
    and for :class:`DeprecationWarning` it is not even *shown* otherwise:
    Python's default filters display one only when it appears to come from
    ``__main__``, so a warning attributed to this module is silently dropped
    in an ordinary script.

    A fixed number cannot do this. The wrapper classes reach the warning
    through two different chains -- most through their own ``__init__`` and
    then ``CustomForce.__init__``, while ``RMSDForce`` and ``RGForce`` have no
    intermediate step -- and a subclass that does not define ``__init__`` at
    all would be shorter again. So count instead of guessing.

    Returns
    -------
    int
        A ``stacklevel`` for :func:`warnings.warn`, where 1 means the caller
        of ``warnings.warn`` itself.
    """
    level = 1
    frame = sys._getframe(1)          # the caller, i.e. stacklevel 1
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _warn_wrapper_deprecated(wrapper_class,
                             ctor_args='"<energy expression>"'):
    """Warn that a pyCHARMM wrapper force class is on its way out.

    The wrapper classes exist because forces used to be built on CHARMM's C++
    side, one hand-written entry point per parameter per force type. A force
    built with the ``openmm`` package and handed over with
    :func:`add_openmm_force` is the same C++ force at the same speed, with the
    whole OpenMM API available instead of the part the wrappers reproduced --
    so the wrappers no longer earn their keep.

    Every wrapper class shares its name with the OpenMM class it stands in
    for, which is what makes the suggested replacement a mechanical
    substitution.

    Parameters
    ----------
    wrapper_class : str
        Name of the deprecated class, which is also the OpenMM class name.
    ctor_args : str, optional
        Exactly what to show between the parentheses of the constructor call
        in the example. Defaults to a placeholder energy expression, which is
        what most of these classes take.

    Warns
    -----
    DeprecationWarning
        Always, and attributed to the caller's line rather than to this
        module -- see :func:`_stacklevel_of_caller`, since that attribution
        is what decides whether it is shown at all. pyCHARMM's own pytest
        configuration ignores these, so this is a message for people reading
        their own scripts' output rather than something that interrupts a run.
    """
    warnings.warn(
        f"omm.{wrapper_class} is deprecated and will be removed in a future "
        f"release. Build the force with the openmm package and hand it to "
        f"CHARMM instead:\n"
        f"    import openmm\n"
        f"    force = openmm.{wrapper_class}({ctor_args})\n"
        f"    index = omm.add_openmm_force(force)\n"
        f"It is the same force at the same speed, and the whole OpenMM API is "
        f"available rather than the part these classes reproduced.",
        DeprecationWarning,
        stacklevel=_stacklevel_of_caller(),
    )


def _require_openmm_build():
    """Raise if this CHARMM was built without OpenMM support.

    In a CHARMM built without OpenMM the underlying entry points are stubs
    that call CHARMM's fatal-error path, which ends the whole process at the
    default bomb level.  This check therefore has to come first: it turns
    what would be an abrupt termination into a Python exception the caller
    can catch, and avoids the misleading "enable OpenMM first" message for a
    build where OpenMM cannot be enabled at all.

    Raises
    ------
    NotImplementedError
        If CHARMM has no OpenMM support compiled in.
    """
    if omm_version() <= 0:
        raise NotImplementedError(
            "This CHARMM was built without OpenMM support, so OpenMM forces "
            "and energy terms are not available. Rebuild CHARMM with OpenMM "
            "enabled to use them."
        )


def _kind_for_force(force, openmm):
    """Return the `CustomForceType` for an OpenMM force object, or None.

    Matches the force's exact class first, then walks its base classes so a
    user-defined subclass of a supported force is still recognised. The most
    derived match wins, so a subclass is never mistaken for a different
    force type.

    Parameters
    ----------
    force : openmm.Force
        The force whose type is wanted.
    openmm : module
        The already-imported ``openmm`` module, passed in so this module does
        not import openmm at import time.

    Returns
    -------
    CustomForceType or None
        The matching force type, or None if this is not a class pyCHARMM can
        add.
    """
    for cls in type(force).__mro__:
        kind = _OPENMM_CLASS_TO_KIND.get(cls.__name__)
        if kind is not None and isinstance(force, getattr(openmm, cls.__name__,
                                                          type(force))):
            return kind
    return None


def _validate_eterm(eterm):
    """Check that `eterm` is a usable CHARMM energy-term selector.

    Parameters
    ----------
    eterm : EtermBucket or int
        The value to check.

    Returns
    -------
    int
        The validated integer term code.

    Raises
    ------
    TypeError
        If `eterm` is not an integer or `EtermBucket`.
    ValueError
        If `eterm` is not one of the `EtermBucket` values. Negative values are
        rejected here as well: only :func:`set_force_eterm` with ``None``
        clears an override, so a stray negative number cannot silently do it.
    """
    if isinstance(eterm, bool) or not isinstance(eterm, (int, EtermBucket)):
        raise TypeError(
            f"energy term must be an EtermBucket or int, got "
            f"{type(eterm).__name__}. Use e.g. omm.EtermBucket.CFEX."
        )
    code = int(eterm)
    valid = [b.value for b in EtermBucket]
    if code not in valid:
        names = ", ".join(f"{b.name}={b.value}" for b in EtermBucket)
        raise ValueError(
            f"{code} is not a valid CHARMM energy term for an OpenMM force. "
            f"Valid terms are: {names}. Pass None to restore the default "
            f"term for the force's class."
        )
    return code


def _check_openmm_build_matches():
    """Verify Python's ``openmm`` and CHARMM's OpenMM are the same build.

    Handing CHARMM a pointer to an object created by a *different* OpenMM
    library is undefined behaviour: the two libraries have separate type
    information and separate heaps, so the pointer is meaningless to CHARMM
    and the run will usually crash while computing energies. Comparing
    versions catches the common cause -- a conda environment whose ``openmm``
    package was upgraded after CHARMM was built.

    Raises
    ------
    ImportError
        If the ``openmm`` Python package is not installed.
    RuntimeError
        If the two OpenMM versions differ, with instructions for fixing the
        environment.

    Notes
    -----
    Matching versions do not *prove* the two are the same shared library,
    only that they are the same release. If you see crashes inside OpenMM
    after adding a force this way, confirm that the ``libOpenMM`` CHARMM was
    linked against is the one in the environment you are running in.
    """
    try:
        import openmm
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "adding a pre-built OpenMM force needs the openmm Python "
            "package in this environment "
            "(conda install -c conda-forge openmm)."
        ) from exc

    charmm_ver = omm_version()                       # e.g. 84 for 8.4

    # CHARMM reports 0 when it has no OpenMM at all: nothing to compare, and
    # the caller has a bigger problem that other calls report already.
    if charmm_ver <= 0:
        # NotImplementedError, matching _require_openmm_build and the rest of
        # the module's missing-feature errors, so one except clause covers
        # every "this build cannot do that" case.
        raise NotImplementedError(
            "This CHARMM was not built with OpenMM support, so OpenMM "
            "forces cannot be added. Rebuild CHARMM with OpenMM enabled."
        )

    py_parts = openmm.__version__.split(".")
    try:
        py_ver = int(py_parts[0]) * 10 + int(py_parts[1])
    except (IndexError, ValueError):                 # pragma: no cover
        return                                       # unparsable; skip check

    if py_ver == charmm_ver:
        return

    c_major, c_minor = divmod(charmm_ver, 10)
    # CHARMM's build-time probe can fail and leave a placeholder version. In
    # that case we cannot tell a real mismatch from a bad probe, so warn
    # instead of refusing to run -- refusing would make the feature unusable
    # in a build that is probably fine.
    if charmm_ver == 60:
        warnings.warn(
            f"Could not confirm the OpenMM build matches: CHARMM reports "
            f"OpenMM {c_major}.{c_minor} (a placeholder its build-time probe "
            f"uses when detection fails) while the openmm Python package is "
            f"{openmm.__version__}. Proceeding. If CHARMM crashes inside "
            f"OpenMM after adding a force, make sure the libOpenMM CHARMM "
            f"was linked against is the one in this environment.",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    raise RuntimeError(
        f"OpenMM version mismatch: the openmm Python package is "
        f"{openmm.__version__} but this CHARMM was built against OpenMM "
        f"{c_major}.{c_minor}. Passing a force between mismatched OpenMM "
        f"builds would crash. Install openmm {c_major}.{c_minor} in this "
        f"environment (conda install -c conda-forge "
        f"'openmm={c_major}.{c_minor}'), or rebuild CHARMM against openmm "
        f"{openmm.__version__}."
    )


def _refuse_unsupported_class(class_name):
    """Raise `TypeError` explaining why this force class cannot be added.

    Two different refusals hide behind one symptom, and telling them apart is
    the whole point of this function. A force CHARMM already builds for itself
    is refused *permanently*, because handing one over would double-count an
    interaction rather than add a new one -- so "not supported yet" would be a
    lie that invites the caller to wait for a release that should never come.
    Anything else is genuinely just not wired up.

    Parameters
    ----------
    class_name : str
        The OpenMM class name of the offered force.

    Raises
    ------
    TypeError
        Always. This function only ever refuses.
    """
    duplicates = _CHARMM_OWNED_CLASSES.get(class_name)
    if duplicates is not None:
        remedy = _CHARMM_OWNED_REMEDY.get(class_name)
        where = (f"Set it through {remedy} instead."
                 if remedy else
                 "Set it through CHARMM instead.")
        raise TypeError(
            f"CHARMM builds {duplicates} itself, so adding an OpenMM "
            f"{class_name} would count {duplicates} twice -- at whatever "
            f"settings this force carries, with nothing to warn you. {where} "
            f"This is not a limitation waiting to be lifted: the two would "
            f"always be additive."
        )
    supported = ", ".join(sorted(_OPENMM_CLASS_TO_KIND))
    raise TypeError(
        f"pyCHARMM cannot add an OpenMM {class_name} yet. Supported force "
        f"classes are: {supported}."
    )


def _warn_about_force_group(force, class_name):
    """Warn if this force carries a force group CHARMM is about to overwrite.

    CHARMM assigns OpenMM force groups itself when it builds the system, one
    per energy term, so that it can report each term separately. A group set
    on the force beforehand is therefore discarded. Staying silent about it
    would leave a caller believing a setting took effect when it did not.

    Only a non-default group is worth mentioning: group 0 is what OpenMM uses
    unless asked otherwise, so it carries no intent.

    Parameters
    ----------
    force : openmm.Force
        The force about to be handed over.
    class_name : str
        Its class name, for the message.

    Warns
    -----
    RuntimeWarning
        If the force's group is not 0.
    """
    try:
        group = force.getForceGroup()
    except Exception:                                # pragma: no cover
        return                                       # not worth failing over
    if group == 0:
        return
    warnings.warn(
        f"This OpenMM {class_name} has force group {group} set, which CHARMM "
        f"will overwrite when it builds the system: it assigns one group per "
        f"CHARMM energy term so each term can be reported separately. To "
        f"choose which term this force reports in, pass eterm= or use "
        f"set_force_eterm().",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_about_custom_nonbonded(class_name):
    """Warn about the two ways a CustomNonbondedForce surprises people.

    It is accepted, and it does work, but on terms worth stating up front:
    OpenMM requires its particle count and exclusions to line up with the
    whole system, which is not how CHARMM builds its nonbonded list, and the
    force *adds to* CHARMM's own nonbonded energy rather than replacing it.
    Both show up later as a confusing exception during system setup or as an
    energy that looks double-counted.

    Parameters
    ----------
    class_name : str
        The force's class name, for the message.

    Warns
    -----
    RuntimeWarning
        Always, when a nonbonded custom force is added.
    """
    warnings.warn(
        f"An OpenMM {class_name} is added on top of CHARMM's own nonbonded "
        f"energy, not in place of it, so both contribute. OpenMM also expects "
        f"its particle count and exclusions to match the whole system, which "
        f"is not how CHARMM builds its nonbonded list; a mismatch surfaces as "
        f"an OpenMM exception while the system is being built, not here.",
        RuntimeWarning,
        stacklevel=3,
    )


# The System a caller handed over, kept alive here.  CHARMM stores only the
# pointer and cannot keep a Python object alive, so without this a caller who
# does not hold their own reference -- omm.use_external_system(openmm.System())
# is the obvious way to write it -- would have the System collected out from
# under CHARMM and segfault on the next energy.  Verified: that is exactly what
# happened before this existed.
_supplied_system_ref = None


def _refuse_if_supplied_system_built(action):
    """Refuse a change that a supplied System can no longer accept.

    Adding a force is fine: it can be put into the System as it stands. What
    cannot is any change to a force already copied into the System -- its
    energy term, its parameters, whether it is switched on -- because CHARMM's
    answer to those is to throw the System away and build another, and it can
    neither destroy nor empty a System it does not own.

    Without this the change is accepted and the *next* energy evaluation ends
    the run from inside CHARMM, at a line the caller never wrote. Refusing here
    keeps the run alive with the physics it already had, and points at the
    thing that has to change.

    Parameters
    ----------
    action : str
        What the caller was trying to do, named in the error.

    Raises
    ------
    RuntimeError
        If a System supplied by the caller has already been built.
    """
    if omm_version() <= 0:
        return
    if _external_system_state() == 2:
        raise RuntimeError(
            f"{action} cannot take effect: the OpenMM System you supplied has "
            f"already been built, and CHARMM cannot add to a System it does "
            f"not own -- rebuilding it would count every particle and force "
            f"twice. Make this change before the first energy evaluation, or "
            f"call omm.clear() and start over with a fresh System."
        )


def _ensure_openmm_numpy_c_api():
    """Make OpenMM's Python module initialise numpy's C API, or say why not.

    Works around an OpenMM bug that segfaults a ``PythonForce`` when CHARMM is
    the one that starts the evaluation. When the callback returns, OpenMM asks
    whether the forces it got back are a numpy array, and asks through numpy's
    C API table -- a variable private to ``openmm._openmm`` that
    ``import_array()`` fills in. Nothing on that path fills it in first. It is
    normally filled already, because most OpenMM Python calls go through an
    argument converter that fills it, which is why the bug is invisible in
    ordinary OpenMM scripts and why the same force works when Python asks for
    the energy. Drive the Context from CHARMM and no such call need ever have
    happened: the table is null, and reading it is a segmentation fault with no
    message. Reproduced in pure OpenMM with no CHARMM in the process, on 8.5.2
    and 8.6.

    So one throwaway call is made here through that converter. It is a plain
    OpenMM Python call on an object nobody else can see, it touches nothing the
    caller owns, and it costs microseconds. Once OpenMM fixes this upstream
    the call becomes harmless rather than wrong, so it is safe to leave.

    Upstream has now fixed it: openmm/openmm#5400, released in **8.6.1**. This
    function and the two calls to it can be deleted once the oldest OpenMM
    pyCHARMM supports is 8.6.1 or newer -- until then 8.5.2 and 8.6.0 still
    need it. ``test_the_python_force_guard_is_still_needed`` in
    ``tests/test_openmm_external_context.py`` is the canary that proves the bug
    is live, and it skips itself from 8.6.1 on.

    Notes
    -----
    Failure is not fatal. If this cannot be done, the only thing lost is the
    workaround: everything except a Python-callback force is unaffected, and
    refusing the handoff over it would break working scripts. A warning is
    issued instead, since the failure it guards against is a crash with no
    output.
    """
    try:
        import openmm
        probe = openmm.CustomExternalForce("0")
        probe.addParticle(0, [])         # the converter that does the work
    except Exception as exc:             # noqa: BLE001 -- must never be fatal
        warnings.warn(
            f"Could not get OpenMM to initialise numpy's C API ({exc!r}). "
            f"Everything here still works except an openmm.PythonForce, "
            f"which may crash the run when CHARMM evaluates it. Calling "
            f"context.setPositions(...) yourself, from Python, avoids that.",
            RuntimeWarning,
            stacklevel=3,
        )


def use_external_system(system):
    """Have CHARMM fill in an OpenMM System you created, instead of its own.

    **When would I want this?** Only as the first step of creating the OpenMM
    Context yourself -- to choose the platform and its properties, to bring
    your own integrator, or to run a force that has to be driven from Python.
    If you do not need the Context, you do not need this: leave CHARMM to make
    its own System, which is what happens by default and needs no setup.

    The reason it works this way round is that a Context must be built on the
    System CHARMM actually evaluates, and CHARMM cannot hand a System outward
    after the fact -- a raw ``OpenMM::System`` pointer cannot be turned back
    into a working Python object. So you make the System, keep it, and let
    CHARMM fill it in.

    What goes into it is still CHARMM's: the particles from the PSF, the
    restraints, any forces you added to CHARMM's store, and the temperature
    and pressure control implied by your dynamics options. You will see all of
    that appear in your own object.

    **Adding your own forces to it.** Do that with ``system.addForce(force)``
    after CHARMM has filled the System in, using a force you have not touched
    the ownership of. Do not set ``force.thisown = False`` first: OpenMM
    refuses such a force with *"the System object does not own its
    corresponding OpenMM object"*, which names the System but is really about
    the force, so the message sends you looking in the wrong place. Handing a
    force to :func:`add_openmm_force` instead is the other way to do this, and
    there ownership is managed for you.

    Add it *before* you create the Context, because a Context is built from
    the System as it stands. A force added afterwards contributes nothing
    until the Context is told about it, and one added after
    :func:`use_external_context` does nothing at all.

    Parameters
    ----------
    system : openmm.System
        A System you created and still hold a reference to. CHARMM fills it in
        and never destroys it.

    Raises
    ------
    ImportError
        If the ``openmm`` Python package is not available.
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    RuntimeError
        If Python's OpenMM and CHARMM's OpenMM are different versions, or if a
        previously supplied System has already been filled in (see Notes).
    TypeError
        If `system` is not an ``openmm.System``.

    Notes
    -----
    You do not have to keep your own reference alive: pyCHARMM holds one
    until the System is released or OpenMM is cleared. CHARMM itself stores
    only the pointer and cannot keep a Python object alive, so without that
    reference the obvious spelling --
    ``omm.use_external_system(openmm.System())`` -- would have the System
    collected while CHARMM was still using it.

    It is filled in **once**. CHARMM's usual answer to a changed force is to
    throw its System away and build another, which it cannot do to a System it
    does not own -- and filling the same one again would add every particle and
    force a second time. So add your forces before the first energy, and if you
    do need to change them afterwards, call :func:`omm.clear` and start over
    with a fresh System.

    Examples
    --------
    ``omm.enable()`` is the step that is easy to miss. Supplying the System
    does not by itself route the energy through OpenMM, so without it the
    ``energy.show()`` below is an ordinary CHARMM energy, nothing is put into
    your System, and :func:`use_external_context` later refuses it with *"the
    System you supplied has not been filled in yet"*.

    >>> import openmm                                     # doctest: +SKIP
    >>> import pycharmm.omm as omm                        # doctest: +SKIP
    >>> from pycharmm import energy                       # doctest: +SKIP
    >>> omm.enable()                                      # doctest: +SKIP
    >>> system = openmm.System()                          # doctest: +SKIP
    >>> omm.use_external_system(system)                   # doctest: +SKIP
    >>> energy.show()             # CHARMM fills it in    # doctest: +SKIP
    >>> system.getNumParticles()                          # doctest: +SKIP
    22
    """
    _check_openmm_build_matches()
    import openmm

    if not isinstance(system, openmm.System):
        raise TypeError(
            f"use_external_system needs an openmm.System, got "
            f"{type(system).__name__}. Create one with openmm.System()."
        )

    # Before anything is handed over: a PythonForce put into this System would
    # otherwise crash the run the first time CHARMM evaluated it.
    _ensure_openmm_numpy_c_api()

    state = _external_system_state()
    if state == 2:
        raise RuntimeError(
            "The System supplied earlier has already been filled in, and "
            "filling one twice would count every particle and force twice. "
            "Call omm.clear() and supply a fresh System to start over."
        )

    lib.api_omm_set_external_system.argtypes = [ctypes.c_void_p]
    lib.api_omm_set_external_system.restype = None
    lib.api_omm_set_external_system(ctypes.c_void_p(int(system.this)))

    # Hold it only after CHARMM has accepted it, so a refused handover does
    # not leave a reference to a System nobody is using.
    global _supplied_system_ref
    _supplied_system_ref = system


def release_external_system():
    """Go back to letting CHARMM create its own OpenMM System.

    **When would I want this?** After :func:`use_external_system`, if you want
    CHARMM to manage things again without clearing everything else. The System
    you supplied is not destroyed -- it is still yours.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    """
    _require_openmm_build()
    lib.api_omm_release_external_system.restype = None
    lib.api_omm_release_external_system()

    global _supplied_system_ref
    _supplied_system_ref = None


def _external_system_state():
    """Return CHARMM's view of an externally supplied System.

    Returns
    -------
    int
        0 when CHARMM is using its own System, 1 when a supplied System is in
        use but not yet filled in, 2 when a supplied System has been filled in
        and so cannot be used again.
    """
    _require_openmm_build()
    lib.api_omm_external_system_state.restype = ctypes.c_int
    return lib.api_omm_external_system_state()


# The Context a caller handed over, kept alive here for the same reason as
# _supplied_system_ref: CHARMM stores only the pointer.
_supplied_context_ref = None

_ADOPT_CONTEXT_ERRORS = {
    1: "no System has been supplied yet. Call omm.use_external_system() "
       "first, so the Context can be built on the System CHARMM fills in.",
    2: "the System you supplied has not been filled in yet, so a Context "
       "built on it would have no particles and no forces. Run an energy "
       "first (energy.show()), then create the Context.",
    3: "the Context was built on a different System. It is missing every "
       "force CHARMM added, so its energies would be wrong rather than "
       "obviously broken. Build the Context on the System you passed to "
       "omm.use_external_system().",
}


def use_external_context(context):
    """Have CHARMM drive an OpenMM Context you created.

    **When would I want this?** To choose the platform and its properties
    yourself, to bring your own integrator, or to run a force that has to be
    driven from Python. If none of those apply, leave CHARMM to make its own
    Context -- that is the default and needs no setup.

    The order matters, because a Context is built on a System and CHARMM has
    to have filled that System in first:

    1. ``omm.enable()`` -- without this the energy in step 3 is an ordinary
       CHARMM energy and your System is never filled in
    2. ``omm.use_external_system(system)``
    3. run an energy, so CHARMM puts the particles and forces into it
    4. create your integrator and ``openmm.Context`` on that same System
    5. ``omm.use_external_context(context)``

    The platform the run uses is the one on your Context, so there is no need
    to match it in step 1.

    CHARMM then pushes positions, velocities, time and box vectors in and
    reads energies and forces back, as it does with its own Context, and
    never destroys it.

    Parameters
    ----------
    context : openmm.Context
        A Context built on the System passed to :func:`use_external_system`.

    Raises
    ------
    ImportError
        If the ``openmm`` Python package is not available.
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    RuntimeError
        If no System was supplied, if it has not been filled in yet, or if
        the Context was built on a different System.
    TypeError
        If `context` is not an ``openmm.Context``.

    Notes
    -----
    **The integrator becomes yours, and so do its consequences.** CHARMM
    steps whatever integrator the Context holds, so the timestep, temperature
    and friction actually used are the ones you set, not the ones on the
    ``DYNAmics`` command. CHARMM's own time accounting still comes from the
    command, so a mismatch there makes the reported time wrong.

    **Each dynamics command starts the integrator from clean state.** CHARMM
    does that for its own Context by rebuilding it, which is not available
    here, so it calls ``reinitialize()`` discarding state instead and pushes
    the run state straight back. This is not cosmetic: with that reset
    removed, the stochastic restart stages of CHARMM's own OpenMM dynamics
    test move by 0.8% to 3.6%, while deterministic ones do not move at all.

    Keep your reference if you like, but you do not have to: pyCHARMM holds
    one until the Context is released or OpenMM is cleared.

    Examples
    --------
    >>> import openmm                                      # doctest: +SKIP
    >>> import pycharmm.omm as omm                         # doctest: +SKIP
    >>> from pycharmm import energy                        # doctest: +SKIP
    >>> omm.enable()                                       # doctest: +SKIP
    >>> system = openmm.System()                           # doctest: +SKIP
    >>> omm.use_external_system(system)                    # doctest: +SKIP
    >>> energy.show()      # CHARMM fills the System in    # doctest: +SKIP
    >>> integrator = openmm.LangevinMiddleIntegrator(      # doctest: +SKIP
    ...     300*openmm.unit.kelvin, 1/openmm.unit.picosecond,
    ...     0.002*openmm.unit.picoseconds)
    >>> ctx = openmm.Context(system, integrator)           # doctest: +SKIP
    >>> omm.use_external_context(ctx)                      # doctest: +SKIP
    """
    _check_openmm_build_matches()
    import openmm

    if not isinstance(context, openmm.Context):
        raise TypeError(
            f"use_external_context needs an openmm.Context, got "
            f"{type(context).__name__}. Create one with openmm.Context("
            f"system, integrator)."
        )

    _ensure_openmm_numpy_c_api()

    lib.api_omm_set_external_context.argtypes = [ctypes.c_void_p]
    lib.api_omm_set_external_context.restype = ctypes.c_int
    status = lib.api_omm_set_external_context(
        ctypes.c_void_p(int(context.this)))
    if status != 0:
        raise RuntimeError(
            "CHARMM will not drive this OpenMM Context: "
            + _ADOPT_CONTEXT_ERRORS.get(status, f"status {status}")
        )

    global _supplied_context_ref
    _supplied_context_ref = context


def release_external_context():
    """Go back to letting CHARMM create its own OpenMM Context.

    **When would I want this?** After :func:`use_external_context`, to hand
    control back without clearing everything else. Neither your Context nor
    your System is destroyed -- both are still yours.

    This gives back **both**. The System was adopted alongside the Context and
    cannot be filled in twice, so CHARMM returns to making its own of each --
    the only state it can build a fresh Context from. To go on supplying the
    System, call :func:`use_external_system` again with a fresh one.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    """
    _require_openmm_build()
    lib.api_omm_release_external_context.restype = None
    lib.api_omm_release_external_context()

    global _supplied_context_ref
    _supplied_context_ref = None


def _external_context_state():
    """Return 1 if CHARMM is driving a Context it does not own, else 0."""
    _require_openmm_build()
    lib.api_omm_external_context_state.restype = ctypes.c_int
    return lib.api_omm_external_context_state()


def get_forces():
    """Return what CHARMM's OpenMM force store currently holds.

    **When to prefer this.** Reach for it whenever an energy is not what you
    expected and OpenMM forces are involved. Until now there was no way to
    ask what CHARMM was holding: you could not count the forces, and the only
    way to learn whether one was switched on was to toggle it and read back
    the previous value. Most confusion in this area is one of three things
    this answers directly -- a force added twice, a force that is switched
    off, or a force reporting in a different energy term than you meant.

    Returns
    -------
    pandas.DataFrame
        One row per stored force, in store-index order, with columns:

        ``index``
            The force's store index, as returned when it was added.
        ``kind``
            Its `CustomForceType`.
        ``eterm``
            The `EtermBucket` it reports in.
        ``eterm_is_default``
            True when that term comes from the force's kind rather than from
            an explicit :func:`set_force_eterm`.
        ``enabled``
            True if the force is switched on. Only enabled forces are copied
            into the OpenMM system, so a disabled one contributes nothing.

        An empty frame with those columns if the store holds nothing.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.

    Notes
    -----
    This describes the store, not the OpenMM system CHARMM is evaluating.
    They differ if a force was added or changed since the system was built --
    which is exactly what :func:`system_changed` and the store's own mutators
    ask CHARMM to put right.

    Examples
    --------
    >>> import pycharmm.omm as omm
    >>> omm.get_forces()                                   # doctest: +SKIP
       index                        kind eterm  eterm_is_default  enabled
    0      0  CustomForceType.EXTERNAL  CFCV             False     True
    1      1      CustomForceType.BOND  CFIN              True    False
    """
    _require_openmm_build()
    import pandas as pd

    columns = ["index", "kind", "eterm", "eterm_is_default", "enabled"]

    lib.api_cf_get_num_forces.restype = ctypes.c_int
    lib.api_cf_get_kind.argtypes = [ctypes.c_int]
    lib.api_cf_get_kind.restype = ctypes.c_int
    lib.api_cf_is_enabled.argtypes = [ctypes.c_int]
    lib.api_cf_is_enabled.restype = ctypes.c_int

    rows = []
    for index in range(lib.api_cf_get_num_forces()):
        raw_kind = lib.api_cf_get_kind(ctypes.c_int(index))
        try:
            kind = CustomForceType(raw_kind)
        except ValueError:                               # pragma: no cover
            # A kind this pyCHARMM does not know about: report it as the raw
            # number rather than dropping the row, so an unexpected force is
            # visible instead of invisible.
            kind = raw_kind
        override = get_force_eterm(index)
        if override is None:
            default = CUSTOM_FORCE_BUCKETS.get(kind)
            eterm = EtermBucket[default] if default else None
            is_default = True
        else:
            eterm = override
            is_default = False
        rows.append({
            "index": index,
            "kind": kind,
            "eterm": eterm,
            "eterm_is_default": is_default,
            "enabled": lib.api_cf_is_enabled(ctypes.c_int(index)) == 1,
        })
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        # EtermBucket is an IntEnum, so pandas takes the column for plain
        # integers and stores 2 where EtermBucket.CFEX was put in. Rebuild it
        # as an object column to keep the members themselves, which is what
        # makes `row.eterm.name` and `is EtermBucket.CFEX` work for callers.
        frame["eterm"] = pd.Series([r["eterm"] for r in rows], dtype=object)
    return frame


def show_forces():
    """Print what CHARMM's OpenMM force store currently holds.

    **When to prefer this.** Use it interactively, when you want to look
    rather than to compute; :func:`get_forces` returns the same thing as a
    table you can test against. This follows the convention elsewhere in
    pyCHARMM, where ``show_*`` prints and ``get_*`` returns.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.

    Examples
    --------
    >>> import pycharmm.omm as omm
    >>> omm.show_forces()                                  # doctest: +SKIP
    index  kind                  eterm  enabled
        0  CustomExternalForce   CFCV   yes
        1  CustomBondForce       CFIN   no      (off: not in the system)
    """
    forces = get_forces()
    if forces.empty:
        print("CHARMM's OpenMM force store is empty.")
        return
    print(f"{'index':>5}  {'kind':<22} {'eterm':<6} {'enabled':<7}")
    for row in forces.itertuples(index=False):
        kind = row.kind.name if hasattr(row.kind, "name") else str(row.kind)
        eterm = row.eterm.name if row.eterm is not None else "?"
        if not row.eterm_is_default:
            eterm += "*"
        note = "" if row.enabled else "  (off: not in the system)"
        print(f"{row.index:>5}  {kind:<22} {eterm:<6} "
              f"{'yes' if row.enabled else 'no':<7}{note}")
    if not forces["eterm_is_default"].all():
        print("* energy term set explicitly, not the default for that kind")


def system_changed():
    """Tell CHARMM that a force it is holding has been changed from outside.

    CHARMM builds its OpenMM system once and reuses it, so it has to be told
    when something in that system is no longer what it copied. Everything in
    pyCHARMM that changes a stored force does this for you.

    **When to prefer this.** Exactly one situation needs it: you handed a
    force to CHARMM with :func:`add_openmm_force` and then changed that same
    object through your own reference to it -- adding a particle, setting a
    global parameter, editing a tabulated function. CHARMM cannot see that
    happen, so without this call the energy silently keeps the value it had
    before your change. You do not need it after
    :func:`add_openmm_force` itself, after :func:`set_force_eterm`, or after
    any wrapper-class method; those already ask for the rebuild.

    Calling it when nothing changed is harmless -- it costs one rebuild of the
    OpenMM system at the next energy or dynamics command.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.

    Examples
    --------
    >>> import openmm
    >>> import pycharmm.omm as omm
    >>> f = openmm.CustomExternalForce("k*x^2")
    >>> _ = f.addGlobalParameter("k", 1.0)
    >>> _ = f.addParticle(0, [])
    >>> i = omm.add_openmm_force(f)               # doctest: +SKIP
    >>> f.setGlobalParameterDefaultValue(0, 5.0)  # doctest: +SKIP
    >>> omm.system_changed()   # without this, k stays 1.0  # doctest: +SKIP
    """
    _require_openmm_build()
    _refuse_if_supplied_system_built("Asking CHARMM to rebuild")
    lib.api_omm_invalidate()


def add_openmm_force(force, eterm=None):
    """Add an OpenMM force object built in Python to CHARMM's OpenMM system.

    Lets you build a force with the full OpenMM Python API and hand the
    finished object to CHARMM, rather than rebuilding it through pyCHARMM's
    own wrapper classes. The force is still evaluated by OpenMM in C++, at
    full speed.

    **This is now the way to add a force.** pyCHARMM's own wrapper classes
    (``omm.CustomExternalForce`` and friends) are deprecated in favour of it
    and will be removed in a future release, so new code should come here.
    Reach for this whenever you want the OpenMM API itself: following OpenMM
    documentation or an example, reusing force-building code from another
    OpenMM project, or needing something the wrappers never exposed. Neither
    route is faster; both end up as the same C++ force inside OpenMM.

    The one requirement is that Python's OpenMM and CHARMM's OpenMM are the
    same library. That is the ordinary case -- conda-forge ships the library
    and the Python module in a single package -- and a version mismatch is
    refused with instructions rather than left to crash.

    CHARMM takes over the force object: Python stops owning it, so do not
    also add it to an OpenMM ``System`` yourself and do not add the same
    object twice (both are refused). Each call needs its own force object.

    Parameters
    ----------
    force : openmm.Force
        A force created with the ``openmm`` Python package, e.g.
        ``openmm.CustomExternalForce("k*x^2")``. Its class must be one of the
        supported types (the keys of `_OPENMM_CLASS_TO_KIND`).
    eterm : EtermBucket or int, optional
        CHARMM energy term to report this force's energy in. The default
        (None) uses the term implied by the force's class, as listed in
        `CUSTOM_FORCE_BUCKETS`.

    Returns
    -------
    int
        The force's index in CHARMM's force store, for use with
        :func:`set_force_eterm` and the other store functions.

    Raises
    ------
    ImportError
        If the ``openmm`` Python package is not available.
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    RuntimeError
        If Python's OpenMM and CHARMM's OpenMM are different versions, or if
        CHARMM rejects the force (CHARMM also prints the reason and the
        remedy to its output).
    TypeError
        If `force` is not an ``openmm.Force``, or is a force class pyCHARMM
        does not support.

    Notes
    -----
    Python and CHARMM must be using the same OpenMM installation. This is
    checked by comparing versions before the force is handed over; see
    :func:`_check_openmm_build_matches`.

    Forces added this way are dropped when OpenMM is cleared (``omm.clear()``
    or ``OMM CLEAR``), and store indices start over afterwards. Add your
    forces again after clearing, and do not hold on to old indices.

    Because CHARMM took over the force object, do not keep using the Python
    object after handing it over -- build a fresh force for each add. Calling
    methods on a handed-over force reaches a C++ object Python no longer
    controls, and after a clear it may no longer be valid at all.

    If CHARMM crashes inside OpenMM shortly after adding a force, the usual
    cause is Python and CHARMM using two different OpenMM libraries even
    though the versions match. Check that the ``libOpenMM`` CHARMM was linked
    against is the one in the environment you are running in.

    Examples
    --------
    Pull one atom along +x, reporting the energy in the external term:

    >>> import openmm
    >>> import pycharmm.omm as omm
    >>> f = openmm.CustomExternalForce("-(fx*x)")
    >>> _ = f.addPerParticleParameter("fx")
    >>> _ = f.addParticle(0, [100.0])
    >>> index = omm.add_openmm_force(f)                 # doctest: +SKIP

    The same force, but reported in the collective-variable term instead:

    >>> index = omm.add_openmm_force(f, eterm=omm.EtermBucket.CFCV)  # doctest: +SKIP
    """
    _check_openmm_build_matches()
    import openmm

    # Guard the C side's precondition: it can narrow a Force to a subclass
    # but cannot tell a Force from an unrelated pointer, so anything that is
    # not a Force must be stopped here.
    if not isinstance(force, openmm.Force):
        raise TypeError(
            f"add_openmm_force needs an openmm.Force, got "
            f"{type(force).__name__}. Build the force with the openmm "
            f"Python package, e.g. openmm.CustomExternalForce('k*x^2')."
        )

    class_name = type(force).__name__
    kind = _kind_for_force(force, openmm)
    if kind is None:
        _refuse_unsupported_class(class_name)

    _warn_about_force_group(force, class_name)
    if kind is CustomForceType.NONBONDED:
        _warn_about_custom_nonbonded(class_name)

    # A force that no longer owns its C++ object has already been given away
    # -- to CHARMM by an earlier call, or to an OpenMM System. Adopting it
    # again would double-count its energy or free it twice.
    if not force.thisown:
        raise RuntimeError(
            f"This OpenMM {class_name} has already been handed to CHARMM or "
            f"added to an OpenMM System, so it cannot be added again. Build "
            f"a new force object for each add_openmm_force call."
        )

    # Validate the requested energy term BEFORE registering the force, so a
    # bad term cannot leave a force registered on the wrong term with no
    # index returned to the caller.
    if eterm is not None:
        _validate_eterm(eterm)

    ptr = int(force.this)
    lib.api_custom_force_add_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.api_custom_force_add_ptr.restype = ctypes.c_int
    index = lib.api_custom_force_add_ptr(ctypes.c_void_p(ptr),
                                         ctypes.c_int(kind.value))
    if index < 0:
        reason = _FSTORE_ERRORS.get(index, f"status {index}")
        raise RuntimeError(
            f"CHARMM would not add this OpenMM {class_name}: {reason} "
            f"The force was not added and its energy will not be included."
        )

    # Only now hand ownership over: if the add had failed, Python must keep
    # owning the object so it is freed normally instead of leaked.
    force.thisown = False

    if eterm is not None:
        set_force_eterm(index, eterm)
    return index


def set_force_eterm(index, eterm):
    """Choose which CHARMM energy term a stored force reports its energy in.

    Overrides the term implied by the force's class. Takes effect the next
    time CHARMM builds its OpenMM system, i.e. at the next energy or dynamics
    command.

    **When to prefer this.** Use it when you want a force's energy reported
    on its own line instead of added in with every other force of the same
    kind -- for example a restraint you want to watch separately from the
    other external forces, or several forces you want separated into
    different terms so you can follow them independently. If you do not care
    which line the energy appears on, leave the default alone. To set the term
    at the same time you add the force, pass ``eterm`` to
    :func:`add_openmm_force` instead of calling this afterwards; for
    pyCHARMM's wrapper classes, :meth:`CustomForce.set_eterm` is the same
    thing spelled on the force object.

    Parameters
    ----------
    index : int
        The force's index in CHARMM's force store, as returned by
        :func:`add_openmm_force` or a `CustomForce` object's ``index``.
    eterm : EtermBucket or int or None
        The energy term to use. Pass None to clear a previous override and go
        back to the default for the force's class.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    RuntimeError
        If CHARMM rejects the request, e.g. the index does not exist or the
        term code is out of range. CHARMM also prints the reason and remedy.

    Examples
    --------
    >>> import pycharmm.omm as omm
    >>> omm.set_force_eterm(0, omm.EtermBucket.CFCV)     # doctest: +SKIP
    >>> omm.set_force_eterm(0, None)   # back to the default  # doctest: +SKIP
    """
    _require_openmm_build()
    _refuse_if_supplied_system_built("Changing a force's energy term")
    code = -1 if eterm is None else _validate_eterm(eterm)
    lib.api_cf_set_bucket.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.api_cf_set_bucket.restype = ctypes.c_int
    status = lib.api_cf_set_bucket(ctypes.c_int(int(index)),
                                   ctypes.c_int(code))
    if status < 0:
        reason = _FSTORE_ERRORS.get(status, f"status {status}")
        raise RuntimeError(
            f"CHARMM would not set the energy term for force {index}: "
            f"{reason} The force keeps its previous energy term."
        )


def get_force_eterm(index):
    """Return the energy-term override set on a stored force, if any.

    **When to prefer this.** Use it to check what a force was pinned to,
    typically when a script sets terms conditionally or when you are working
    out why an energy is showing up on an unexpected line. It reports only
    what was explicitly set: a force using the default term for its class
    returns None, and `CUSTOM_FORCE_BUCKETS` tells you what that default is.
    OpenMM force groups are not the thing to inspect here: CHARMM assigns
    those itself while building the system, so the energy term is what a
    caller actually controls.

    Parameters
    ----------
    index : int
        The force's index in CHARMM's force store.

    Returns
    -------
    EtermBucket or None
        The override set by :func:`set_force_eterm`, or None if the force
        uses the default term for its class.

    Raises
    ------
    NotImplementedError
        If this CHARMM was built without OpenMM support.
    IndexError
        If there is no stored force with that index. (CHARMM reports "no
        override" and "no such force" the same way, so the index is checked
        here first rather than reporting a missing force as a default term.)

    Examples
    --------
    >>> import pycharmm.omm as omm
    >>> omm.get_force_eterm(0)                           # doctest: +SKIP
    <EtermBucket.CFCV: 4>
    """
    _require_openmm_build()
    lib.api_cf_get_bucket.argtypes = [ctypes.c_int]
    lib.api_cf_get_bucket.restype = ctypes.c_int
    code = lib.api_cf_get_bucket(ctypes.c_int(int(index)))
    if code == -5:                        # FSTORE_ERR_BAD_INDEX
        raise IndexError(
            f"there is no stored OpenMM force with index {index}. Add the "
            f"force first; add_openmm_force returns its index."
        )
    if code == -1:                        # no override set for this force
        return None
    if code < 0:
        # Any other negative status is a failure, not "no override"; saying
        # "uses the default term" here would hide it.
        raise RuntimeError(
            f"CHARMM could not report the energy term for force {index} "
            f"(status {code})."
        )
    try:
        return EtermBucket(code)
    except ValueError:                                   # pragma: no cover
        return None


# ============================================================
# Standalone functions (backward compatibility)
# ============================================================

def custom_force_add(kind, energy, n=-1):
    """Create a custom force of `kind` in CHARMM and return its index.

    Low-level helper behind the `CustomForce` classes.  In user code prefer
    those classes, or `add_openmm_force`.

    Parameters
    ----------
    kind : CustomForceType or int
        Which kind of OpenMM custom force to create.
    energy : str
        OpenMM energy expression, in OpenMM units (nm, kJ/mol).
    n : int, optional
        Particle or group count for the kinds that need one (compound bond,
        centroid bond, many-particle).  Default -1, meaning unused.

    Returns
    -------
    int
        Index of the new force in CHARMM's force store.
    """
    c_energy = ctypes.create_string_buffer(str.encode(energy))
    c_kind = ctypes.c_int(kind.value if isinstance(kind, CustomForceType)
                          else int(kind))
    c_n = ctypes.c_int(n)
    new_index = lib.api_custom_force_add(c_kind, c_energy, c_n)
    return new_index


def customnb_add_force(energy):
    """Create a CustomNonbondedForce and return its index.

    Thin wrapper kept for older scripts; `CustomNonbondedForce` is the
    documented way to do this.

    Parameters
    ----------
    energy : str
        OpenMM energy expression, in OpenMM units.

    Returns
    -------
    int
        Index of the new force in CHARMM's force store.
    """
    new_index = custom_force_add(CustomForceType.NONBONDED, energy)
    return new_index


def customnb_add_particle(force_index, params):
    """Add one particle's parameter values to a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    params : sequence of float
        Per-particle parameter values, in the order the parameters were
        declared.

    Returns
    -------
    int
        Index of the particle within the force.
    """
    f_i = ctypes.c_int(force_index)
    n = len(params)
    n_params = ctypes.c_int(n)
    c_params = (ctypes.c_double * n)(*params)
    new_index = lib.api_customnb_add_particle(f_i, c_params, n_params)
    return new_index


def customnb_add_particle_param(force_index, param_name):
    """Declare a per-particle parameter on a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    param_name : str
        Parameter name as used in the energy expression.

    Returns
    -------
    int
        Index of the new parameter.
    """
    i = ctypes.c_int(force_index)
    c_name = ctypes.create_string_buffer(str.encode(param_name))
    new_part_param_i = lib.api_customnb_add_particle_param(i, c_name)
    return new_part_param_i


def customnb_add_global_param(force_index, param_name, param_value):
    """Declare a global parameter on a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    param_name : str
        Parameter name as used in the energy expression.
    param_value : float
        Default value for the parameter.

    Returns
    -------
    int
        Index of the new parameter.
    """
    i = ctypes.c_int(force_index)
    c_name = ctypes.create_string_buffer(str.encode(param_name))
    c_val = ctypes.c_double(param_value)
    new_i = lib.api_customnb_add_global_param(i, c_name, c_val)
    return new_i


def customnb_set_global_param(force_index, param_index, param_value):
    """Set the default value of a CustomNonbondedForce global parameter.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    param_index : int
        Index of the parameter, from `customnb_add_global_param`.
    param_value : float
        New default value.
    """
    f_i = ctypes.c_int(force_index)
    p_i = ctypes.c_int(param_index)
    c_val = ctypes.c_double(param_value)
    lib.api_customnb_set_global_param(f_i, p_i, c_val)


def customnb_add_exclusion(force_index, particle1_index, particle2_index):
    """Exclude one pair of particles from a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    particle1_index, particle2_index : int
        The two particles whose interaction is skipped.

    Returns
    -------
    int
        Index of the new exclusion.
    """
    f_i = ctypes.c_int(force_index)
    p1_i = ctypes.c_int(particle1_index)
    p2_i = ctypes.c_int(particle2_index)
    new_ex_i = lib.api_customnb_add_exclusion(f_i, p1_i, p2_i)
    return new_ex_i


def customnb_change_nonbonded_method(force_index, method_index):
    """Set the nonbonded method of a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    method_index : int
        OpenMM nonbonded method code (for example NoCutoff or
        CutoffPeriodic).
    """
    f_i = ctypes.c_int(force_index)
    m_i = ctypes.c_int(method_index)
    lib.api_customnb_change_nonbonded_method(f_i, m_i)


def customnb_change_cutoff(force_index, new_cutoff):
    """Set the cutoff distance of a CustomNonbondedForce.

    Parameters
    ----------
    force_index : int
        Index of the force in CHARMM's force store.
    new_cutoff : float
        Cutoff distance in nm (OpenMM units).
    """
    f_i = ctypes.c_int(force_index)
    cutoff = ctypes.c_double(new_cutoff)
    lib.api_customnb_change_cutoff(f_i, cutoff)


# ============================================================
# Class-based API
# ============================================================

def _c_str(s):
    """Convert a Python string to a null-terminated ctypes buffer."""
    return ctypes.create_string_buffer(s.encode())


def _c_double_array(vals):
    """Convert a sequence of floats to a ctypes double array."""
    n = len(vals)
    return (ctypes.c_double * n)(*vals), ctypes.c_int(n)


def _c_int_array(vals):
    """Convert a sequence of ints to a ctypes int array."""
    n = len(vals)
    return (ctypes.c_int * n)(*vals), ctypes.c_int(n)


class CustomForce:
    """Base class for all OpenMM Custom*Force types stored in CHARMM's
    ForcesStore.

    Energy expressions use OpenMM units (nm, kJ/mol, ps) since OpenMM
    parses them directly.
    """

    _kind = None  # Subclasses must set this

    def _deprecation_ctor_args(self):
        """What this class's OpenMM constructor takes, for the example.

        Most of these forces take only an energy expression. The ones whose
        OpenMM constructor differs override this, so the replacement shown in
        the deprecation warning is a call that actually works rather than one
        the user has to correct.
        """
        return '"<energy expression>"'

    def __init__(self, energy_expression, n=-1):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        n : int, optional
            Particle or group count, for the subclasses that need one.
            Default -1, meaning unused.

        Raises
        ------
        NotImplementedError
            If this CHARMM was built without OpenMM support.
        TypeError
            If called on `CustomForce` itself rather than a subclass.
        RuntimeError
            If CHARMM could not create the force.

        Notes
        -----
        The build check comes first and is what makes this safe to call on a
        CHARMM built without OpenMM. Without it the call reaches a Fortran
        stub that raises a CHARMM fatal error, which at the default bomb
        level ends the whole process -- so a script could not catch the
        problem, or even report it, because it was already gone.

        Warns
        -----
        DeprecationWarning
            These classes are being retired in favour of
            :func:`add_openmm_force`. See :func:`_warn_wrapper_deprecated`.
        """
        _require_openmm_build()
        kind = self._kind
        if kind is None:
            # Rejected before warning: there is no openmm.CustomForce to
            # recommend, so a deprecation notice here would name a class that
            # does not exist, on a path that is already an error.
            raise TypeError("Cannot instantiate CustomForce directly")
        _warn_wrapper_deprecated(self.__class__.__name__,
                                 self._deprecation_ctor_args())
        c_energy = _c_str(energy_expression)
        c_kind = ctypes.c_int(kind.value)
        c_n = ctypes.c_int(n)
        self._index = lib.api_custom_force_add(c_kind, c_energy, c_n)
        if self._index < 0:
            raise RuntimeError(
                f"Failed to create {self.__class__.__name__}")

    @property
    def index(self):
        """The force's index in the ForcesStore (0-based)."""
        return self._index

    def set_eterm(self, eterm):
        """Choose which CHARMM energy term this force reports its energy in.

        Convenience wrapper around :func:`set_force_eterm` for this force.

        **When to prefer this.** Use it on a force you built with a pyCHARMM
        wrapper class, whenever you want that force's energy on its own line
        rather than summed with other forces of the same kind. This is also
        the replacement for :meth:`set_force_group`, which cannot work because
        CHARMM assigns OpenMM force groups itself.

        Parameters
        ----------
        eterm : EtermBucket or int or None
            The energy term to use, or None to restore the default for this
            force's class.

        Examples
        --------
        >>> import pycharmm.omm as omm
        >>> f = omm.CustomExternalForce("k*x^2")         # doctest: +SKIP
        >>> f.set_eterm(omm.EtermBucket.CFCV)            # doctest: +SKIP
        """
        set_force_eterm(self._index, eterm)

    def get_eterm(self):
        """Return this force's energy-term override, or None if unset.

        **When to prefer this.** Use it to confirm what this force was pinned
        to, or to tell "explicitly set" apart from "using the class default"
        (None). It is :meth:`get_force_group`'s useful counterpart: the force
        group is reassigned by CHARMM, whereas the energy term is what you
        actually control.

        Returns
        -------
        EtermBucket or None
            The override set by :meth:`set_eterm`, or None if this force uses
            the default term for its class.
        """
        return get_force_eterm(self._index)

    def add_global_parameter(self, name, default_value):
        """Add a global parameter to the force expression.

        Returns the parameter index."""
        c_name = _c_str(name)
        c_val = ctypes.c_double(default_value)
        return lib.api_cf_add_global_param(
            ctypes.c_int(self._index), c_name, c_val)

    def set_global_parameter_default_value(self, param_index, value):
        """Set the default value of a global parameter."""
        lib.api_cf_set_global_param(
            ctypes.c_int(self._index),
            ctypes.c_int(param_index),
            ctypes.c_double(value))

    def get_num_global_parameters(self):
        """Return the number of global parameters."""
        return lib.api_cf_get_num_global_params(
            ctypes.c_int(self._index))

    def get_global_parameter_default_value(self, param_index):
        """Get the default value of a global parameter."""
        lib.api_cf_get_global_param_default_value.restype = \
            ctypes.c_double
        return lib.api_cf_get_global_param_default_value(
            ctypes.c_int(self._index), ctypes.c_int(param_index))

    def add_energy_parameter_derivative(self, name):
        """Request derivative of energy with respect to a global parameter.

        Not supported by CustomExternalForce, CustomHbondForce,
        or CustomManyParticleForce."""
        c_name = _c_str(name)
        lib.api_cf_add_energy_param_deriv(
            ctypes.c_int(self._index), c_name)

    def set_uses_periodic_boundary_conditions(self, periodic):
        """Set whether this force uses periodic boundary conditions."""
        lib.api_cf_set_uses_pbc(
            ctypes.c_int(self._index),
            ctypes.c_int(1 if periodic else 0))

    def turn_on(self):
        """Enable this force for the next OpenMM context creation."""
        return force_turn_on(self._index)

    def turn_off(self):
        """Disable this force for the next OpenMM context creation."""
        return force_turn_off(self._index)

    # ---- Force groups ----

    def set_force_group(self, group):
        """Set the OpenMM force group (0-31) for this force.

        .. deprecated::
            Setting a force group has no lasting effect. CHARMM assigns force
            groups itself when it builds the OpenMM system, one per energy
            term, so that it can report each term separately; whatever is set
            here is overwritten at that point.

            To control which CHARMM energy term this force contributes to,
            use :meth:`set_eterm` instead.

        Parameters
        ----------
        group : int
            OpenMM force group, 0-31.

        Warns
        -----
        DeprecationWarning
            Always, because the setting does not survive system build.
        """
        warnings.warn(
            "set_force_group has no lasting effect: CHARMM assigns OpenMM "
            "force groups itself when it builds the system, one per energy "
            "term, and overwrites whatever is set here. Use set_eterm() to "
            "choose which CHARMM energy term this force reports in.",
            DeprecationWarning,
            stacklevel=2,
        )
        lib.api_cf_set_force_group(
            ctypes.c_int(self._index), ctypes.c_int(group))

    def get_force_group(self):
        """Return the force group currently stored on this force object.

        Notes
        -----
        This is the value on the stored force, which is **not** the group
        OpenMM ends up using: CHARMM assigns groups from its energy-term
        mapping when it builds the system. Use :meth:`get_eterm` to find out
        which CHARMM energy term this force reports in.

        Returns
        -------
        int
            The force group stored on the force object.
        """
        return lib.api_cf_get_force_group(
            ctypes.c_int(self._index))

    # ---- Introspection ----

    def get_energy_expression(self):
        """Return the energy expression string."""
        buf = ctypes.create_string_buffer(1024)
        lib.api_cf_get_energy_expression(
            ctypes.c_int(self._index), buf, ctypes.c_int(1024))
        return buf.value.decode()

    def get_global_parameter_name(self, param_index):
        """Return the name of global parameter at the given index."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_get_global_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()

    def get_num_per_parameters(self):
        """Return the number of per-particle/bond parameters.

        For CustomHbondForce, returns the number of per-donor parameters."""
        return lib.api_cf_get_num_per_params(
            ctypes.c_int(self._index))

    def get_per_parameter_name(self, param_index):
        """Return the name of per-particle/bond parameter at the given index.

        For CustomHbondForce, returns per-donor parameter names."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_get_per_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()

    # Force types that support addTabulatedFunction
    _TABULATED_TYPES = {
        CustomForceType.NONBONDED, CustomForceType.COMPOUND_BOND,
        CustomForceType.CENTROID_BOND, CustomForceType.GB,
        CustomForceType.H_BOND, CustomForceType.MANY_PARTICLE,
        CustomForceType.CV,
    }

    def add_tabulated_function_continuous1d(self, name, values,
                                            min_val, max_val, periodic=False):
        """Add a Continuous1DFunction tabulated function.

        Supported by: Nonbonded, CompoundBond, CentroidBond, GB,
                      Hbond, ManyParticle, CV force types."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, c_n = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_continuous1d(
            ctypes.c_int(self._index), c_name, c_vals, c_n,
            ctypes.c_double(min_val), ctypes.c_double(max_val),
            ctypes.c_int(1 if periodic else 0))

    def add_tabulated_function_discrete1d(self, name, values):
        """Add a Discrete1DFunction tabulated function.

        Supported by: Nonbonded, CompoundBond, CentroidBond, GB,
                      Hbond, ManyParticle, CV force types."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, c_n = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_discrete1d(
            ctypes.c_int(self._index), c_name, c_vals, c_n)

    def add_tabulated_function_continuous2d(self, name, values,
                                            nx, ny,
                                            xmin, xmax, ymin, ymax,
                                            periodic=False):
        """Add a Continuous2DFunction tabulated function.

        values is a flat list of nx*ny values in row-major order."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, _ = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_continuous2d(
            ctypes.c_int(self._index), c_name, c_vals,
            ctypes.c_int(nx), ctypes.c_int(ny),
            ctypes.c_double(xmin), ctypes.c_double(xmax),
            ctypes.c_double(ymin), ctypes.c_double(ymax),
            ctypes.c_int(1 if periodic else 0))

    def add_tabulated_function_discrete2d(self, name, values, nx, ny):
        """Add a Discrete2DFunction tabulated function.

        values is a flat list of nx*ny values in row-major order."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, _ = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_discrete2d(
            ctypes.c_int(self._index), c_name, c_vals,
            ctypes.c_int(nx), ctypes.c_int(ny))

    def add_tabulated_function_continuous3d(self, name, values,
                                            nx, ny, nz,
                                            xmin, xmax, ymin, ymax,
                                            zmin, zmax, periodic=False):
        """Add a Continuous3DFunction tabulated function.

        values is a flat list of nx*ny*nz values."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, _ = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_continuous3d(
            ctypes.c_int(self._index), c_name, c_vals,
            ctypes.c_int(nx), ctypes.c_int(ny), ctypes.c_int(nz),
            ctypes.c_double(xmin), ctypes.c_double(xmax),
            ctypes.c_double(ymin), ctypes.c_double(ymax),
            ctypes.c_double(zmin), ctypes.c_double(zmax),
            ctypes.c_int(1 if periodic else 0))

    def add_tabulated_function_discrete3d(self, name, values, nx, ny, nz):
        """Add a Discrete3DFunction tabulated function.

        values is a flat list of nx*ny*nz values."""
        if self._kind not in self._TABULATED_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "tabulated functions")
        c_name = _c_str(name)
        c_vals, _ = _c_double_array(values)
        return lib.api_cf_add_tabulated_function_discrete3d(
            ctypes.c_int(self._index), c_name, c_vals,
            ctypes.c_int(nx), ctypes.c_int(ny), ctypes.c_int(nz))

    # Force types that support updateParametersInContext
    _UPDATE_TYPES = {
        CustomForceType.BOND, CustomForceType.ANGLE,
        CustomForceType.TORSION, CustomForceType.EXTERNAL,
        CustomForceType.NONBONDED, CustomForceType.COMPOUND_BOND,
        CustomForceType.CENTROID_BOND, CustomForceType.GB,
        CustomForceType.H_BOND, CustomForceType.MANY_PARTICLE,
    }

    def update_parameters_in_context(self):
        """Push modified parameters to a live OpenMM context.

        Call this after using set_*_parameters methods to apply changes
        to a running simulation. Supported by most force types except
        CustomCVForce and CustomVolumeForce."""
        if self._kind not in self._UPDATE_TYPES:
            raise TypeError(
                f"{self.__class__.__name__} does not support "
                "updateParametersInContext")
        lib.api_cf_update_parameters_in_context(
            ctypes.c_int(self._index))


class CustomBondForce(CustomForce):
    """A custom bond force between pairs of particles.

    The energy expression can use the variable 'r' for the bond distance."""

    _kind = CustomForceType.BOND

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_bond_parameter(self, name):
        """Add a per-bond parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_bond_add_per_bond_param(
            ctypes.c_int(self._index), c_name)

    def add_bond(self, particle1, particle2, parameters=None):
        """Add a bond between two particles. Returns the bond index.

        Parameters should match the per-bond parameters in order."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_bond_add_bond(
            ctypes.c_int(self._index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            c_params, c_n)

    def get_num_bonds(self):
        """Return the number of bonds."""
        return lib.api_cf_bond_get_num_bonds(
            ctypes.c_int(self._index))

    def set_bond_parameters(self, index, particle1, particle2, parameters):
        """Set the parameters of a bond."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_bond_set_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            c_params, c_n)

    def get_bond_parameters(self, index, num_parameters):
        """Get the parameters of a bond.

        Returns (particle1, particle2, [param1, param2, ...])."""
        p1 = ctypes.c_int()
        p2 = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_bond_get_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(p1), ctypes.byref(p2),
            params, ctypes.c_int(num_parameters))
        return (p1.value, p2.value, list(params))


class CustomAngleForce(CustomForce):
    """A custom angle force between triplets of particles.

    The energy expression can use the variable 'theta' for the angle."""

    _kind = CustomForceType.ANGLE

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_angle_parameter(self, name):
        """Add a per-angle parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_angle_add_per_angle_param(
            ctypes.c_int(self._index), c_name)

    def add_angle(self, particle1, particle2, particle3, parameters=None):
        """Add an angle between three particles. Returns the angle index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_angle_add_angle(
            ctypes.c_int(self._index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            ctypes.c_int(particle3),
            c_params, c_n)

    def get_num_angles(self):
        """Return the number of angles."""
        return lib.api_cf_angle_get_num_angles(
            ctypes.c_int(self._index))

    def set_angle_parameters(self, index, particle1, particle2, particle3,
                             parameters):
        """Set the parameters of an angle."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_angle_set_angle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            ctypes.c_int(particle3), c_params, c_n)

    def get_angle_parameters(self, index, num_parameters):
        """Get the parameters of an angle.

        Returns (particle1, particle2, particle3, [param1, ...])."""
        p1 = ctypes.c_int()
        p2 = ctypes.c_int()
        p3 = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_angle_get_angle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(p1), ctypes.byref(p2), ctypes.byref(p3),
            params, ctypes.c_int(num_parameters))
        return (p1.value, p2.value, p3.value, list(params))


class CustomTorsionForce(CustomForce):
    """A custom torsion (dihedral) force between quadruplets of particles.

    The energy expression can use the variable 'theta' for the torsion angle."""

    _kind = CustomForceType.TORSION

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_torsion_parameter(self, name):
        """Add a per-torsion parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_torsion_add_per_torsion_param(
            ctypes.c_int(self._index), c_name)

    def add_torsion(self, particle1, particle2, particle3, particle4,
                    parameters=None):
        """Add a torsion between four particles. Returns the torsion index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_torsion_add_torsion(
            ctypes.c_int(self._index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            ctypes.c_int(particle3), ctypes.c_int(particle4),
            c_params, c_n)

    def get_num_torsions(self):
        """Return the number of torsions."""
        return lib.api_cf_torsion_get_num_torsions(
            ctypes.c_int(self._index))

    def set_torsion_parameters(self, index, particle1, particle2,
                               particle3, particle4, parameters):
        """Set the parameters of a torsion."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_torsion_set_torsion_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(particle1), ctypes.c_int(particle2),
            ctypes.c_int(particle3), ctypes.c_int(particle4),
            c_params, c_n)

    def get_torsion_parameters(self, index, num_parameters):
        """Get the parameters of a torsion.

        Returns (particle1, particle2, particle3, particle4, [param1, ...])."""
        p1 = ctypes.c_int()
        p2 = ctypes.c_int()
        p3 = ctypes.c_int()
        p4 = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_torsion_get_torsion_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(p1), ctypes.byref(p2),
            ctypes.byref(p3), ctypes.byref(p4),
            params, ctypes.c_int(num_parameters))
        return (p1.value, p2.value, p3.value, p4.value, list(params))


class CustomExternalForce(CustomForce):
    """A custom external force applied to individual particles.

    The energy expression can use x, y, z for particle coordinates."""

    _kind = CustomForceType.EXTERNAL

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_particle_parameter(self, name):
        """Add a per-particle parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_external_add_per_particle_param(
            ctypes.c_int(self._index), c_name)

    def add_particle(self, particle, parameters=None):
        """Add a particle to the force. Returns the particle index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_external_add_particle(
            ctypes.c_int(self._index), ctypes.c_int(particle),
            c_params, c_n)

    def get_num_particles(self):
        """Return the number of particles."""
        return lib.api_cf_external_get_num_particles(
            ctypes.c_int(self._index))

    def set_particle_parameters(self, index, particle, parameters):
        """Set the parameters of a particle."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_external_set_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(particle), c_params, c_n)

    def get_particle_parameters(self, index, num_parameters):
        """Get the parameters of a particle.

        Returns (particle, [param1, param2, ...])."""
        particle = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_external_get_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(particle),
            params, ctypes.c_int(num_parameters))
        return (particle.value, list(params))


class CustomNonbondedForce(CustomForce):
    """A custom nonbonded force between all particle pairs.

    The energy expression can use r for the inter-particle distance."""

    _kind = CustomForceType.NONBONDED

    # Mirror OpenMM's NonbondedMethod enum
    NoCutoff = 0
    CutoffNonPeriodic = 1
    CutoffPeriodic = 2

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_particle_parameter(self, name):
        """Add a per-particle parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_nb_add_per_particle_param(
            ctypes.c_int(self._index), c_name)

    def add_particle(self, parameters=None):
        """Add a particle. Returns the particle index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_nb_add_particle(
            ctypes.c_int(self._index), c_params, c_n)

    def add_exclusion(self, particle1, particle2):
        """Add an exclusion between two particles. Returns the exclusion index."""
        return lib.api_cf_nb_add_exclusion(
            ctypes.c_int(self._index),
            ctypes.c_int(particle1), ctypes.c_int(particle2))

    def set_nonbonded_method(self, method):
        """Set the nonbonded method (NoCutoff=0, CutoffNonPeriodic=1,
        CutoffPeriodic=2)."""
        lib.api_cf_nb_set_nonbonded_method(
            ctypes.c_int(self._index), ctypes.c_int(method))

    def set_cutoff_distance(self, cutoff):
        """Set the cutoff distance in nm."""
        lib.api_cf_nb_set_cutoff(
            ctypes.c_int(self._index), ctypes.c_double(cutoff))

    def set_use_switching_function(self, use):
        """Enable or disable the switching function."""
        lib.api_cf_nb_set_use_switching_function(
            ctypes.c_int(self._index), ctypes.c_int(1 if use else 0))

    def set_switching_distance(self, distance):
        """Set the switching distance in nm."""
        lib.api_cf_nb_set_switching_distance(
            ctypes.c_int(self._index), ctypes.c_double(distance))

    def set_cutoff_distance_angstrom(self, cutoff_a):
        """Set the cutoff distance in Angstroms (converts to nm)."""
        self.set_cutoff_distance(cutoff_a * NM_PER_ANGSTROM)

    def set_switching_distance_angstrom(self, distance_a):
        """Set the switching distance in Angstroms (converts to nm)."""
        self.set_switching_distance(distance_a * NM_PER_ANGSTROM)

    def get_num_particles(self):
        """Return the number of particles."""
        return lib.api_cf_nb_get_num_particles(
            ctypes.c_int(self._index))

    def set_particle_parameters(self, index, parameters):
        """Set the parameters of a particle."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_nb_set_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_params, c_n)

    def get_particle_parameters(self, index, num_parameters):
        """Get the parameters of a particle.

        Returns [param1, param2, ...]."""
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_nb_get_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            params, ctypes.c_int(num_parameters))
        return list(params)

    def get_nonbonded_method(self):
        """Get the nonbonded method."""
        return lib.api_cf_nb_get_nonbonded_method(
            ctypes.c_int(self._index))

    def get_cutoff_distance(self):
        """Get the cutoff distance in nm."""
        lib.api_cf_nb_get_cutoff.restype = ctypes.c_double
        return lib.api_cf_nb_get_cutoff(
            ctypes.c_int(self._index))

    def add_interaction_group(self, set1, set2):
        """Add an interaction group. Returns the group index.

        set1 and set2 are sequences of particle indices."""
        c_set1, c_n1 = _c_int_array(set1)
        c_set2, c_n2 = _c_int_array(set2)
        return lib.api_cf_nb_add_interaction_group(
            ctypes.c_int(self._index), c_set1, c_n1, c_set2, c_n2)


class CustomCompoundBondForce(CustomForce):
    """A custom force between groups of particles (compound bonds).

    Constructor requires the number of particles per bond."""

    _kind = CustomForceType.COMPOUND_BOND

    def _deprecation_ctor_args(self):
        """This force's OpenMM constructor takes a count first."""
        return '<particles per bond>, "<energy expression>"'

    def __init__(self, num_particles, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        num_particles : int
            Number of particles per bond.
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression, n=num_particles)

    def add_per_bond_parameter(self, name):
        """Add a per-bond parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_compound_add_per_bond_param(
            ctypes.c_int(self._index), c_name)

    def add_bond(self, particles, parameters=None):
        """Add a compound bond. Returns the bond index.

        particles is a sequence of particle indices."""
        if parameters is None:
            parameters = []
        c_parts, c_np = _c_int_array(particles)
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_compound_add_bond(
            ctypes.c_int(self._index), c_parts, c_np, c_params, c_n)

    def get_num_bonds(self):
        """Return the number of bonds."""
        return lib.api_cf_compound_get_num_bonds(
            ctypes.c_int(self._index))

    def set_bond_parameters(self, index, particles, parameters):
        """Set the parameters of a compound bond."""
        c_parts, c_np = _c_int_array(particles)
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_compound_set_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_parts, c_np, c_params, c_n)

    def get_bond_parameters(self, index, num_particles, num_parameters):
        """Get the parameters of a compound bond.

        Returns ([particle1, ...], [param1, ...])."""
        particles = (ctypes.c_int * num_particles)()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_compound_get_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            particles, ctypes.c_int(num_particles),
            params, ctypes.c_int(num_parameters))
        return (list(particles), list(params))


class CustomCentroidBondForce(CustomForce):
    """A custom force between centroids of groups of particles.

    Constructor requires the number of groups per bond."""

    _kind = CustomForceType.CENTROID_BOND

    def _deprecation_ctor_args(self):
        """This force's OpenMM constructor takes a count first."""
        return '<groups per bond>, "<energy expression>"'

    def __init__(self, num_groups, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        num_groups : int
            Number of particle groups per bond.
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression, n=num_groups)

    def add_per_bond_parameter(self, name):
        """Add a per-bond parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_centroid_add_per_bond_param(
            ctypes.c_int(self._index), c_name)

    def add_group(self, particles, weights=None):
        """Add a group of particles. Returns the group index.

        weights is optional; if None, equal weights are used."""
        if weights is None:
            weights = []
        c_parts, c_np = _c_int_array(particles)
        c_weights, c_nw = _c_double_array(weights)
        return lib.api_cf_centroid_add_group(
            ctypes.c_int(self._index), c_parts, c_np, c_weights, c_nw)

    def add_bond(self, groups, parameters=None):
        """Add a bond between group centroids. Returns the bond index."""
        if parameters is None:
            parameters = []
        c_groups, c_ng = _c_int_array(groups)
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_centroid_add_bond(
            ctypes.c_int(self._index), c_groups, c_ng, c_params, c_n)

    def get_num_groups(self):
        """Return the number of groups."""
        return lib.api_cf_centroid_get_num_groups(
            ctypes.c_int(self._index))

    def get_num_bonds(self):
        """Return the number of bonds."""
        return lib.api_cf_centroid_get_num_bonds(
            ctypes.c_int(self._index))

    def set_group_parameters(self, index, particles, weights=None):
        """Set the parameters of a group."""
        if weights is None:
            weights = []
        c_parts, c_np = _c_int_array(particles)
        c_weights, c_nw = _c_double_array(weights)
        lib.api_cf_centroid_set_group_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_parts, c_np, c_weights, c_nw)

    def get_group_parameters(self, index, max_particles=64, max_weights=64):
        """Get the parameters of a group.

        Returns ([particle1, ...], [weight1, ...])."""
        particles = (ctypes.c_int * max_particles)()
        weights = (ctypes.c_double * max_weights)()
        num_p = ctypes.c_int()
        num_w = ctypes.c_int()
        lib.api_cf_centroid_get_group_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            particles, ctypes.c_int(max_particles),
            weights, ctypes.c_int(max_weights),
            ctypes.byref(num_p), ctypes.byref(num_w))
        return (list(particles[:num_p.value]),
                list(weights[:num_w.value]))

    def set_bond_parameters(self, index, groups, parameters):
        """Set the parameters of a bond."""
        c_groups, c_ng = _c_int_array(groups)
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_centroid_set_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_groups, c_ng, c_params, c_n)

    def get_bond_parameters(self, index, max_groups=8, max_parameters=16):
        """Get the parameters of a bond.

        Returns ([group1, ...], [param1, ...])."""
        groups = (ctypes.c_int * max_groups)()
        params = (ctypes.c_double * max_parameters)()
        num_g = ctypes.c_int()
        num_p = ctypes.c_int()
        lib.api_cf_centroid_get_bond_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            groups, ctypes.c_int(max_groups),
            params, ctypes.c_int(max_parameters),
            ctypes.byref(num_g), ctypes.byref(num_p))
        return (list(groups[:num_g.value]),
                list(params[:num_p.value]))

    def get_num_per_bond_parameters(self):
        """Return the number of per-bond parameters."""
        return lib.api_cf_centroid_get_num_per_bond_params(
            ctypes.c_int(self._index))

    def get_per_bond_parameter_name(self, param_index):
        """Return the name of per-bond parameter at the given index."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_centroid_get_per_bond_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()


class CustomGBForce(CustomForce):
    """A custom generalized Born force.

    Unlike other custom forces, the constructor does not take an energy
    expression. Instead, use add_computed_value() and add_energy_term()."""

    _kind = CustomForceType.GB

    # ComputationType enum
    SingleParticle = 0
    ParticlePair = 1
    ParticlePairNoExclusions = 2

    # NonbondedMethod enum
    NoCutoff = 0
    CutoffNonPeriodic = 1
    CutoffPeriodic = 2

    def _deprecation_ctor_args(self):
        """openmm.CustomGBForce takes no constructor arguments.

        Its energy comes from addEnergyTerm(), not from the
        constructor, so showing a placeholder expression here would
        suggest a call that does not exist.
        """
        return ""

    def __init__(self):
        """Create the force in CHARMM's force store.

        Takes no energy expression: a CustomGBForce is built up from computed
        values and energy terms added afterwards.
        """
        # CustomGBForce constructor ignores the energy expression
        super().__init__("")

    def add_per_particle_parameter(self, name):
        """Add a per-particle parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_gb_add_per_particle_param(
            ctypes.c_int(self._index), c_name)

    def add_particle(self, parameters=None):
        """Add a particle. Returns the particle index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_gb_add_particle(
            ctypes.c_int(self._index), c_params, c_n)

    def add_computed_value(self, name, expression, computation_type):
        """Add a computed value. Returns the computed value index.

        computation_type: SingleParticle=0, ParticlePair=1,
                         ParticlePairNoExclusions=2"""
        c_name = _c_str(name)
        c_expr = _c_str(expression)
        return lib.api_cf_gb_add_computed_value(
            ctypes.c_int(self._index), c_name, c_expr,
            ctypes.c_int(computation_type))

    def add_energy_term(self, expression, computation_type):
        """Add an energy term. Returns the energy term index."""
        c_expr = _c_str(expression)
        return lib.api_cf_gb_add_energy_term(
            ctypes.c_int(self._index), c_expr,
            ctypes.c_int(computation_type))

    def set_nonbonded_method(self, method):
        """Set the nonbonded method."""
        lib.api_cf_gb_set_nonbonded_method(
            ctypes.c_int(self._index), ctypes.c_int(method))

    def set_cutoff_distance(self, cutoff):
        """Set the cutoff distance in nm."""
        lib.api_cf_gb_set_cutoff(
            ctypes.c_int(self._index), ctypes.c_double(cutoff))

    def set_cutoff_distance_angstrom(self, cutoff_a):
        """Set the cutoff distance in Angstroms (converts to nm)."""
        self.set_cutoff_distance(cutoff_a * NM_PER_ANGSTROM)

    def get_num_particles(self):
        """Return the number of particles."""
        return lib.api_cf_gb_get_num_particles(
            ctypes.c_int(self._index))

    def set_particle_parameters(self, index, parameters):
        """Set the parameters of a particle."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_gb_set_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_params, c_n)

    def get_particle_parameters(self, index, num_parameters):
        """Get the parameters of a particle.

        Returns [param1, param2, ...]."""
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_gb_get_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            params, ctypes.c_int(num_parameters))
        return list(params)


class CustomHbondForce(CustomForce):
    """A custom hydrogen bond force between donor and acceptor groups.

    Each donor consists of up to 3 particles (d1, d2, d3) and each
    acceptor consists of up to 3 particles (a1, a2, a3). Use -1 for
    unused particle slots."""

    _kind = CustomForceType.H_BOND

    # NonbondedMethod enum
    NoCutoff = 0
    CutoffNonPeriodic = 1
    CutoffPeriodic = 2

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_per_donor_parameter(self, name):
        """Add a per-donor parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_hbond_add_per_donor_param(
            ctypes.c_int(self._index), c_name)

    def add_per_acceptor_parameter(self, name):
        """Add a per-acceptor parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_hbond_add_per_acceptor_param(
            ctypes.c_int(self._index), c_name)

    def add_donor(self, d1, d2=-1, d3=-1, parameters=None):
        """Add a donor group. Returns the donor index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_hbond_add_donor(
            ctypes.c_int(self._index),
            ctypes.c_int(d1), ctypes.c_int(d2), ctypes.c_int(d3),
            c_params, c_n)

    def add_acceptor(self, a1, a2=-1, a3=-1, parameters=None):
        """Add an acceptor group. Returns the acceptor index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_hbond_add_acceptor(
            ctypes.c_int(self._index),
            ctypes.c_int(a1), ctypes.c_int(a2), ctypes.c_int(a3),
            c_params, c_n)

    def add_exclusion(self, donor, acceptor):
        """Add an exclusion between a donor and acceptor."""
        return lib.api_cf_hbond_add_exclusion(
            ctypes.c_int(self._index),
            ctypes.c_int(donor), ctypes.c_int(acceptor))

    def set_nonbonded_method(self, method):
        """Set the nonbonded method."""
        lib.api_cf_hbond_set_nonbonded_method(
            ctypes.c_int(self._index), ctypes.c_int(method))

    def set_cutoff_distance(self, cutoff):
        """Set the cutoff distance in nm."""
        lib.api_cf_hbond_set_cutoff(
            ctypes.c_int(self._index), ctypes.c_double(cutoff))

    def set_cutoff_distance_angstrom(self, cutoff_a):
        """Set the cutoff distance in Angstroms (converts to nm)."""
        self.set_cutoff_distance(cutoff_a * NM_PER_ANGSTROM)

    def get_num_donors(self):
        """Return the number of donors."""
        return lib.api_cf_hbond_get_num_donors(
            ctypes.c_int(self._index))

    def get_num_acceptors(self):
        """Return the number of acceptors."""
        return lib.api_cf_hbond_get_num_acceptors(
            ctypes.c_int(self._index))

    def set_donor_parameters(self, index, d1, d2, d3, parameters):
        """Set the parameters of a donor."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_hbond_set_donor_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(d1), ctypes.c_int(d2), ctypes.c_int(d3),
            c_params, c_n)

    def get_donor_parameters(self, index, num_parameters):
        """Get the parameters of a donor.

        Returns (d1, d2, d3, [param1, ...])."""
        d1 = ctypes.c_int()
        d2 = ctypes.c_int()
        d3 = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_hbond_get_donor_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(d1), ctypes.byref(d2), ctypes.byref(d3),
            params, ctypes.c_int(num_parameters))
        return (d1.value, d2.value, d3.value, list(params))

    def set_acceptor_parameters(self, index, a1, a2, a3, parameters):
        """Set the parameters of an acceptor."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_hbond_set_acceptor_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.c_int(a1), ctypes.c_int(a2), ctypes.c_int(a3),
            c_params, c_n)

    def get_acceptor_parameters(self, index, num_parameters):
        """Get the parameters of an acceptor.

        Returns (a1, a2, a3, [param1, ...])."""
        a1 = ctypes.c_int()
        a2 = ctypes.c_int()
        a3 = ctypes.c_int()
        params = (ctypes.c_double * num_parameters)()
        lib.api_cf_hbond_get_acceptor_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            ctypes.byref(a1), ctypes.byref(a2), ctypes.byref(a3),
            params, ctypes.c_int(num_parameters))
        return (a1.value, a2.value, a3.value, list(params))

    def get_num_per_donor_parameters(self):
        """Return the number of per-donor parameters."""
        return lib.api_cf_hbond_get_num_per_donor_params(
            ctypes.c_int(self._index))

    def get_num_per_acceptor_parameters(self):
        """Return the number of per-acceptor parameters."""
        return lib.api_cf_hbond_get_num_per_acceptor_params(
            ctypes.c_int(self._index))

    def get_per_donor_parameter_name(self, param_index):
        """Return the name of per-donor parameter at the given index."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_hbond_get_per_donor_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()

    def get_per_acceptor_parameter_name(self, param_index):
        """Return the name of per-acceptor parameter at the given index."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_hbond_get_per_acceptor_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()


class CustomManyParticleForce(CustomForce):
    """A custom many-particle interaction force.

    Constructor requires the number of particles per interaction."""

    _kind = CustomForceType.MANY_PARTICLE

    # NonbondedMethod enum
    NoCutoff = 0
    CutoffNonPeriodic = 1
    CutoffPeriodic = 2

    def _deprecation_ctor_args(self):
        """This force's OpenMM constructor takes a count first."""
        return '<particles per set>, "<energy expression>"'

    def __init__(self, num_particles, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        num_particles : int
            Number of particles per interaction.
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression, n=num_particles)

    def add_per_particle_parameter(self, name):
        """Add a per-particle parameter. Returns the parameter index."""
        c_name = _c_str(name)
        return lib.api_cf_many_add_per_particle_param(
            ctypes.c_int(self._index), c_name)

    def add_particle(self, parameters=None, particle_type=0):
        """Add a particle. Returns the particle index."""
        if parameters is None:
            parameters = []
        c_params, c_n = _c_double_array(parameters)
        return lib.api_cf_many_add_particle(
            ctypes.c_int(self._index), c_params, c_n,
            ctypes.c_int(particle_type))

    def add_exclusion(self, particle1, particle2):
        """Add an exclusion. Returns the exclusion index."""
        return lib.api_cf_many_add_exclusion(
            ctypes.c_int(self._index),
            ctypes.c_int(particle1), ctypes.c_int(particle2))

    def set_nonbonded_method(self, method):
        """Set the nonbonded method."""
        lib.api_cf_many_set_nonbonded_method(
            ctypes.c_int(self._index), ctypes.c_int(method))

    def set_cutoff_distance(self, cutoff):
        """Set the cutoff distance in nm."""
        lib.api_cf_many_set_cutoff(
            ctypes.c_int(self._index), ctypes.c_double(cutoff))

    def set_cutoff_distance_angstrom(self, cutoff_a):
        """Set the cutoff distance in Angstroms (converts to nm)."""
        self.set_cutoff_distance(cutoff_a * NM_PER_ANGSTROM)

    # PermutationMode enum
    SinglePermutation = 0
    UniqueCentralParticle = 1

    def get_num_particles(self):
        """Return the number of particles."""
        return lib.api_cf_many_get_num_particles(
            ctypes.c_int(self._index))

    def set_particle_parameters(self, index, parameters, particle_type=0):
        """Set the parameters and type of a particle."""
        c_params, c_n = _c_double_array(parameters)
        lib.api_cf_many_set_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            c_params, c_n, ctypes.c_int(particle_type))

    def get_particle_parameters(self, index, num_parameters):
        """Get the parameters and type of a particle.

        Returns ([param1, ...], particle_type)."""
        params = (ctypes.c_double * num_parameters)()
        ptype = ctypes.c_int()
        lib.api_cf_many_get_particle_parameters(
            ctypes.c_int(self._index), ctypes.c_int(index),
            params, ctypes.c_int(num_parameters), ctypes.byref(ptype))
        return (list(params), ptype.value)

    def set_type_filter(self, particle_index, types):
        """Set the type filter for a particle slot.

        types is a sequence of allowed particle type integers."""
        c_types, c_n = _c_int_array(types)
        lib.api_cf_many_set_type_filter(
            ctypes.c_int(self._index), ctypes.c_int(particle_index),
            c_types, c_n)

    def get_type_filter(self, particle_index, max_types=64):
        """Get the type filter for a particle slot.

        Returns a list of allowed particle type integers."""
        types = (ctypes.c_int * max_types)()
        num_types = ctypes.c_int()
        lib.api_cf_many_get_type_filter(
            ctypes.c_int(self._index), ctypes.c_int(particle_index),
            types, ctypes.c_int(max_types), ctypes.byref(num_types))
        return list(types[:num_types.value])

    def get_permutation_mode(self):
        """Get the permutation mode (0=SinglePermutation,
        1=UniqueCentralParticle)."""
        return lib.api_cf_many_get_permutation_mode(
            ctypes.c_int(self._index))

    def set_permutation_mode(self, mode):
        """Set the permutation mode."""
        lib.api_cf_many_set_permutation_mode(
            ctypes.c_int(self._index), ctypes.c_int(mode))

    def get_num_per_particle_parameters(self):
        """Return the number of per-particle parameters."""
        return lib.api_cf_many_get_num_per_particle_params(
            ctypes.c_int(self._index))

    def get_per_particle_parameter_name(self, param_index):
        """Return the name of per-particle parameter at the given index."""
        buf = ctypes.create_string_buffer(256)
        lib.api_cf_many_get_per_particle_param_name(
            ctypes.c_int(self._index), ctypes.c_int(param_index),
            buf, ctypes.c_int(256))
        return buf.value.decode()


class CustomCVForce(CustomForce):
    """A force whose energy depends on collective variables.

    Each collective variable is itself a Force object (from the store)."""

    _kind = CustomForceType.CV

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        super().__init__(energy_expression)

    def add_collective_variable(self, name, force):
        """Add a collective variable defined by another force.

        force can be a CustomForce instance (uses its index) or an int
        (direct store index). The force is copied into the CV force.
        Returns the collective variable index."""
        if isinstance(force, int):
            store_index = force
        elif hasattr(force, 'index'):
            store_index = force.index
        else:
            store_index = int(force)
        c_name = _c_str(name)
        return lib.api_cf_cv_add_collective_variable(
            ctypes.c_int(self._index),
            ctypes.c_int(store_index),
            c_name)


class CustomVolumeForce(CustomForce):
    """A custom volume-dependent force (requires OpenMM 8.4+).

    Currently a placeholder with only base class methods."""

    _kind = CustomForceType.VOLUME

    def __init__(self, energy_expression):
        """Create the force in CHARMM's force store.

        Parameters
        ----------
        energy_expression : str
            OpenMM energy expression, in OpenMM units (nm, kJ/mol, ps).
        """
        requires_omm_version(84, "CustomVolumeForce")
        super().__init__(energy_expression)


class RMSDForce:
    """RMSD collective variable force.

    Computes the RMSD of selected particles from a set of reference
    positions.  Intended for use as a collective variable in
    CustomCVForce.

    Parameters
    ----------
    reference_positions : array_like
        Reference positions, shape (natom, 3), in nanometers.
    particles : list of int, optional
        0-based particle indices.  If empty, all particles are used.
    """

    def __init__(self, reference_positions, particles=None):
        """Create an RMSD force against a reference structure.

        Parameters
        ----------
        reference_positions : array_like
            Reference coordinates, shape (natom, 3), in nanometers (OpenMM
            units), matching the class docstring above.
        particles : sequence of int, optional
            Particles to include in the RMSD.  Default None, meaning all
            particles.

        Raises
        ------
        NotImplementedError
            If this CHARMM was built without OpenMM support.  Checked before
            anything else, because the underlying entry point is a stub in
            that build and ends the process rather than returning.
        ValueError
            If `reference_positions` is not shape (natom, 3).
        RuntimeError
            If CHARMM could not create the force.
        """
        _require_openmm_build()
        _warn_wrapper_deprecated(
            "RMSDForce", "<reference positions>, <particles>")
        import numpy as np
        ref = np.asarray(reference_positions, dtype=np.float64)
        if ref.ndim != 2 or ref.shape[1] != 3:
            raise ValueError("reference_positions must be shape (natom, 3)")
        natom = ref.shape[0]
        c_ref = (ctypes.c_double * (natom * 3))(*ref.ravel())
        if particles is None:
            particles = []
        c_parts = (ctypes.c_int * len(particles))(*particles)
        self._index = lib.api_fstore_add_rmsd(
            c_ref, ctypes.c_int(natom),
            c_parts, ctypes.c_int(len(particles)))
        if self._index < 0:
            raise RuntimeError("Failed to create RMSDForce")

    @property
    def index(self):
        """int : The force's index in CHARMM's force store (0-based)."""
        return self._index

    def set_reference_positions(self, reference_positions):
        """Update the reference positions (nanometers)."""
        import numpy as np
        ref = np.asarray(reference_positions, dtype=np.float64)
        natom = ref.shape[0]
        c_ref = (ctypes.c_double * (natom * 3))(*ref.ravel())
        lib.api_cf_rmsd_set_reference_positions(
            ctypes.c_int(self._index), c_ref, ctypes.c_int(natom))

    def set_particles(self, particles):
        """Update the particle indices (0-based)."""
        c_parts = (ctypes.c_int * len(particles))(*particles)
        lib.api_cf_rmsd_set_particles(
            ctypes.c_int(self._index), c_parts,
            ctypes.c_int(len(particles)))


class RGForce:
    """Radius of gyration collective variable force.

    Computes the radius of gyration of selected particles.
    Intended for use as a collective variable in CustomCVForce.

    Parameters
    ----------
    particles : list of int, optional
        0-based particle indices.  If empty, all particles are used.
    """

    def __init__(self, particles=None):
        """Create a radius-of-gyration force.

        Requires OpenMM 8.4 or later; raises NotImplementedError on older
        builds, naming the version needed.

        Parameters
        ----------
        particles : sequence of int, optional
            Particles included in the radius of gyration.  Default None,
            meaning all particles.
        """
        requires_omm_version(84, "RGForce")
        _warn_wrapper_deprecated("RGForce", "<particles>")
        if particles is None:
            particles = []
        c_parts = (ctypes.c_int * len(particles))(*particles)
        self._index = lib.api_fstore_add_rg(
            c_parts, ctypes.c_int(len(particles)))
        if self._index < 0:
            raise RuntimeError("Failed to create RGForce")

    @property
    def index(self):
        """int : The force's index in CHARMM's force store (0-based)."""
        return self._index


# ============================================================
# Torch force functions
# ============================================================

def has_ommtorch():
    """Return True if this CHARMM build supports OpenMM-Torch.

    Builds without ``--with-ommtorch`` route every ``api_torch_*`` call
    into a Fortran stub that calls ``wrndie(-1, ...)``, which terminates
    the process via ``_gfortran_exit``.  In a pytest run that exit
    bypasses session finalization and produces a silent crash with no
    JUnit XML.  Use this helper to guard call sites and tests.
    """
    lib.api_has_ommtorch.restype = ctypes.c_int
    return bool(lib.api_has_ommtorch())


def _require_ommtorch():
    """Raise if this CHARMM was built without OpenMM-Torch support.

    Call at the start of any wrapper that needs the Torch plugin, so the user
    gets a clear message naming the missing build option instead of a failure
    deeper in the C layer.

    Raises
    ------
    NotImplementedError
        If OpenMM-Torch support is not compiled in.
    """
    if not has_ommtorch():
        raise NotImplementedError(
            "This CHARMM build does not support OpenMM-Torch. "
            "Reconfigure with --with-ommtorch and rebuild."
        )


def omm_version():
    """OpenMM version this CHARMM was built against, as ``major*10 + minor``.

    Returns ``0`` when CHARMM was built without OpenMM.  Otherwise
    matches CMake's ``OMM_VER`` (e.g. OpenMM 8.2 -> 82, 8.4 -> 84).
    Use this to gate features that require a minimum OpenMM version.
    """
    lib.api_omm_version.restype = ctypes.c_int
    return int(lib.api_omm_version())


def requires_omm_version(min_ver, feature):
    """Raise ``NotImplementedError`` if linked OpenMM is below ``min_ver``.

    Call from the entry point of any wrapper that exercises an OpenMM
    API which appeared in a specific release (e.g. ``CustomVolumeForce``
    requires OpenMM 8.4).  The error message names ``feature`` and the
    required and actual versions so the user knows what to upgrade.

    A build with no OpenMM at all is reported as such first, rather than as
    an out-of-date OpenMM.  Both end in ``NotImplementedError``, but only one
    of them is fixed by upgrading a package: telling someone whose CHARMM has
    no OpenMM support to "upgrade openmm and rebuild" sends them to change a
    version that was never the problem.

    Raises
    ------
    NotImplementedError
        If this CHARMM has no OpenMM support, or its OpenMM is older than
        `min_ver`.
    """
    _require_openmm_build()
    actual = omm_version()
    if actual < min_ver:
        major, minor = divmod(min_ver, 10)
        a_major, a_minor = divmod(actual, 10)
        raise NotImplementedError(
            f"{feature} requires OpenMM {major}.{minor} or later, but "
            f"this CHARMM was built against OpenMM {a_major}.{a_minor}. "
            f"Upgrade openmm in your environment "
            f"(e.g. conda install -c conda-forge 'openmm>={major}.{minor}') "
            f"and rebuild CHARMM."
        )


def torch_add_force(module_filename):
    """Add a torch force to CHARMM's OpenMM system

    The module filename should be the full path to
    a torch module file on disk.

    Parameters
    ----------
    filename       : str or bytes
                     path to an on disk torch module file
    """
    _require_ommtorch()
    c_filename = ctypes.create_string_buffer(str.encode(module_filename))
    new_force_index = lib.api_torch_add_force(c_filename)
    return new_force_index


def torch_outputs_forces(index):
    """Tell CHARMM this torch force returns forces as well as an energy.

    Set this when the TorchScript module returns a (energy, forces) pair, so
    OpenMM uses the returned forces instead of differentiating the energy.

    Parameters
    ----------
    index : int
        Index of the torch force in CHARMM's force store.
    """
    _require_ommtorch()
    index = ctypes.c_int(index)
    lib.api_torch_outputs_forces(index)


def torch_uses_periodic(index):
    """Tell CHARMM this torch force expects periodic box vectors.

    Parameters
    ----------
    index : int
        Index of the torch force in CHARMM's force store.
    """
    _require_ommtorch()
    index = ctypes.c_int(index)
    lib.api_torch_uses_periodic(index)


def torch_add_global_param(force_index, param_name, param_value):
    """Declare a global parameter on a torch force.

    Parameters
    ----------
    force_index : int
        Index of the torch force in CHARMM's force store.
    param_name : str
        Parameter name, passed through to the TorchScript module.
    param_value : float
        Default value.

    Returns
    -------
    int
        Index of the new parameter.
    """
    _require_ommtorch()
    i = ctypes.c_int(force_index)
    c_name = ctypes.create_string_buffer(str.encode(param_name))
    c_val = ctypes.c_double(param_value)
    new_i = lib.api_torch_add_global_param(i, c_name, c_val)
    return new_i


def torch_set_global_param(force_index, param_index, param_value):
    """Set the default value of a torch force global parameter.

    This changes the value the parameter starts from when the OpenMM context
    is next built; it does not alter a context that already exists.

    Parameters
    ----------
    force_index : int
        Index of the torch force in CHARMM's force store.
    param_index : int
        Index of the parameter, from `torch_add_global_param`.
    param_value : float
        New default value.
    """
    _require_ommtorch()
    f_i = ctypes.c_int(force_index)
    p_i = ctypes.c_int(param_index)
    c_val = ctypes.c_double(param_value)
    lib.api_torch_set_global_param(f_i, p_i, c_val)
