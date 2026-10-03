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

"""DOMDEC (Domain Decomposition) interface for pycharmm.

This module provides functions to enable, configure, and control DOMDEC
parallel domain decomposition for molecular dynamics and energy calculations.

Functions
=========
- `enable` -- Enable DOMDEC for parallel calculations
- `disable` -- Disable DOMDEC
- `is_enabled` -- Check if DOMDEC is currently enabled
- `check_restraints` -- Check active restraints for DOMDEC compatibility

CHARMM Command Reference
========================
- `energy domdec ...` → `domdec.enable(...)`
- DOMDEC options map to function parameters

Restraint Compatibility
=======================
DOMDEC supports ALL standard restraint types:
- Harmonic restraints (CONS HARM) - Full support
- Fix constraints (CONS FIX) - Full support
- Dihedral restraints (CONS DIHE) - Full support
- NOE/PNOE - Full support
- RESD - Full support
- IC restraints - Full support
- Droplet restraints - Full support

Examples
========
>>> import pycharmm.domdec as domdec
>>> import pycharmm.restraints as restraints

# Enable DOMDEC with automatic restraint check
>>> domdec.enable(nproc=4)

# Check if DOMDEC is enabled
>>> domdec.is_enabled()
True

# Disable DOMDEC
>>> domdec.disable()
"""

import warnings
from pycharmm.lingo import charmm_script
import pycharmm.restraints as restraints


# Module-level state
_domdec_enabled = False
_domdec_config = {}


class DomdecRestraintWarning(UserWarning):
    """Warning issued when active restraints may have issues with DOMDEC."""
    pass


class DomdecEngineError(Exception):
    """Error related to DOMDEC engine operations."""
    pass


def check_restraints():
    """Check if active restraints are compatible with DOMDEC.

    DOMDEC supports all standard restraint types, so this typically
    returns an empty list. This function is provided for API consistency
    with blade.py.

    Returns
    -------
    list[str]
        List of incompatible restraint type names, typically empty for DOMDEC.

    Examples
    --------
    >>> import pycharmm.domdec as domdec
    >>> issues = domdec.check_restraints()
    >>> if issues:
    ...     print(f"Incompatible restraints: {issues}")
    """
    from pycharmm.restraints import _state

    conflicts = _state._check_backend_conflicts('domdec')
    return conflicts


def _check_crystal_compatibility():
    """Reject crystal types that the current DOMDEC backend cannot evaluate."""
    import pycharmm.crystal as crystal

    crystal_type = crystal.get_crystal_type_direct()
    if crystal_type is None or not crystal.is_orthorhombic(crystal_type):
        raise DomdecEngineError(
            "DOMDEC requires a defined orthorhombic crystal "
            f"(CUBI, TETR, ORTH, RECT); got {crystal_type or 'none'}."
        )


def enable(gpu=None, gpuid=None, dlb=None, ndir=None, split=None,
           ppang=None, single=False, double=False, test=False,
           warn_restraints=True, **kwargs):
    """Enable DOMDEC for parallel energy and dynamics calculations.

    Parameters
    ----------
    gpu : str or bool, optional
        GPU mode: 'on', 'off', True (='on'), False (='off'), or None.
        - 'on'/True: Enable GPU acceleration
        - 'off'/False: Disable GPU (CPU only)
        - None: Don't specify (use default)
    gpuid : int, optional
        Manually set GPU ID (0 to N-1). -1 for automatic selection.
        None to not specify (uses automatic selection).
    dlb : bool, optional
        Enable/disable dynamic load balancing. None to not specify.
        DLB is ON by default.
    ndir : tuple of (int, int, int), optional
        Spatial division (NX, NY, NZ) for domain decomposition.
        Divides simulation box into NX x NY x NZ sub-boxes.
        If not specified, CHARMM will auto-determine based on CPU count.
    split : bool, optional
        Enable/disable direct/reciprocal split. Turning off can give
        better performance with small CPU counts. None to not specify.
    ppang : int, optional
        Points per Angstrom for lookup tables. Default 200.
        Higher values give more precision but slower simulation.
    single : bool, optional
        If True, perform force calculation in single precision.
        For FFTW users: requires single-precision FFTW library.
    double : bool, optional
        If True, perform force calculation in double precision (default).
    test : bool, optional
        If True, run DOMDEC unit tests.
    warn_restraints : bool, optional
        If True (default), issue a warning if any restraint issues detected.
    **kwargs : dict
        Additional DOMDEC options passed to CHARMM command.

    Notes
    -----
    Equivalent CHARMM commands:
    - `domdec` - Enable with auto settings
    - `domdec gpu on` - Enable with GPU
    - `domdec gpu off` - CPU only
    - `domdec gpuid 0` - Use specific GPU
    - `domdec dlb on/off` - Dynamic load balancing
    - `domdec ndir 2 2 2` - Spatial division
    - `domdec split on/off` - Direct/reciprocal split
    - `domdec ppang 200` - Lookup table precision
    - `domdec single` - Single precision forces
    - `domdec double` - Double precision forces
    - `domdec test` - Run unit tests

    Examples
    --------
    >>> import pycharmm.domdec as domdec

    # Enable with default settings
    >>> domdec.enable()

    # Enable with GPU acceleration
    >>> domdec.enable(gpu=True)

    # Enable with specific GPU
    >>> domdec.enable(gpu=True, gpuid=0)

    # Enable with spatial division
    >>> domdec.enable(ndir=(2, 2, 2))

    # Enable with dynamic load balancing disabled
    >>> domdec.enable(dlb=False)

    # Enable in single precision for performance
    >>> domdec.enable(gpu=True, single=True)
    """
    global _domdec_enabled, _domdec_config

    _check_crystal_compatibility()

    # Check for incompatible restraints (unlikely for DOMDEC but check anyway)
    conflicts = check_restraints()
    if conflicts and warn_restraints:
        msg = (
            f"Active restraints with potential DOMDEC issues: {', '.join(conflicts)}. "
            f"This is unexpected as DOMDEC supports most restraint types."
        )
        warnings.warn(msg, DomdecRestraintWarning, stacklevel=2)

    # Update restraints state to track backend
    restraints.set_backend('domdec')

    # Build DOMDEC command
    cmd_parts = ["domdec"]

    # Handle NDIR option (spatial division)
    if ndir is not None:
        if len(ndir) != 3:
            raise ValueError("ndir must be a tuple of (nx, ny, nz)")
        nx, ny, nz = ndir
        cmd_parts.append(f"ndir {nx} {ny} {nz}")

    # Handle GPU option
    if gpu is not None:
        if isinstance(gpu, bool):
            cmd_parts.append(f"gpu {'on' if gpu else 'off'}")
        elif isinstance(gpu, str):
            cmd_parts.append(f"gpu {gpu}")

    # Handle GPUID option
    if gpuid is not None:
        cmd_parts.append(f"gpuid {gpuid}")

    # Handle DLB option
    if dlb is not None:
        cmd_parts.append(f"dlb {'on' if dlb else 'off'}")

    # Handle SPLIT option
    if split is not None:
        cmd_parts.append(f"split {'on' if split else 'off'}")

    # Handle PPANG option
    if ppang is not None:
        cmd_parts.append(f"ppang {ppang}")

    # Handle precision options
    if single:
        cmd_parts.append("single")
    elif double:
        cmd_parts.append("double")

    # Handle TEST option
    if test:
        cmd_parts.append("test")

    # Add any additional kwargs
    for key, value in kwargs.items():
        if isinstance(value, bool):
            cmd_parts.append(f"{key} {'on' if value else 'off'}")
        else:
            cmd_parts.append(f"{key} {value}")

    # Execute DOMDEC command
    cmd = " ".join(cmd_parts)
    charmm_script(cmd)

    _domdec_enabled = True
    _domdec_config = {
        'gpu': gpu,
        'gpuid': gpuid,
        'dlb': dlb,
        'ndir': ndir,
        'split': split,
        'ppang': ppang,
        'single': single,
        'double': double,
        **kwargs
    }


def disable():
    """Disable DOMDEC and return to standard calculations.

    Notes
    -----
    Equivalent CHARMM command: `energy domdec off` or similar

    Examples
    --------
    >>> import pycharmm.domdec as domdec
    >>> domdec.disable()
    """
    global _domdec_enabled, _domdec_config

    charmm_script("energy domdec off")
    _domdec_enabled = False
    _domdec_config = {}

    # Reset backend to standard
    restraints.set_backend('standard')


def is_enabled():
    """Check if DOMDEC is currently enabled.

    Returns
    -------
    bool
        True if DOMDEC is enabled, False otherwise.

    Examples
    --------
    >>> import pycharmm.domdec as domdec
    >>> domdec.enable()
    >>> domdec.is_enabled()
    True
    >>> domdec.disable()
    >>> domdec.is_enabled()
    False
    """
    return _domdec_enabled


def get_config():
    """Get current DOMDEC configuration.

    Returns
    -------
    dict
        Dictionary with current DOMDEC configuration, empty if not enabled.

    Examples
    --------
    >>> import pycharmm.domdec as domdec
    >>> domdec.enable(gpu=True, dlb=False)
    >>> domdec.get_config()
    {'gpu': True, 'dlb': False}
    """
    return _domdec_config.copy()


def energy(gpu=None, gpuid=None, dlb=None, ndir=None, split=None,
           ppang=None, single=False, double=False, show=True, **kwargs):
    """Calculate energy using DOMDEC.

    Parameters
    ----------
    gpu : str or bool, optional
        GPU mode: 'on', 'off', True, False, or None.
    gpuid : int, optional
        Manually set GPU ID (0 to N-1). -1 for automatic selection.
    dlb : bool, optional
        Enable/disable dynamic load balancing.
    ndir : tuple of (int, int, int), optional
        Spatial division (NX, NY, NZ) for domain decomposition.
    split : bool, optional
        Enable/disable direct/reciprocal split.
    ppang : int, optional
        Points per Angstrom for lookup tables.
    single : bool, optional
        If True, perform force calculation in single precision.
    double : bool, optional
        If True, perform force calculation in double precision.
    show : bool, optional
        If True (default), display energy breakdown.
    **kwargs : dict
        Additional options passed to CHARMM command.

    Notes
    -----
    Equivalent CHARMM command: `energy domdec [options]`

    Examples
    --------
    >>> import pycharmm.domdec as domdec

    # Energy with DOMDEC GPU
    >>> domdec.energy(gpu=True, dlb=False)

    # Energy with DOMDEC CPU only
    >>> domdec.energy(gpu=False, dlb=False)

    # Energy with specific GPU
    >>> domdec.energy(gpu=True, gpuid=0)

    # Energy with spatial division
    >>> domdec.energy(ndir=(2, 2, 2))
    """
    global _domdec_enabled

    _check_crystal_compatibility()

    # Build energy command
    cmd_parts = ["energy domdec"]

    # Handle NDIR option
    if ndir is not None:
        if len(ndir) != 3:
            raise ValueError("ndir must be a tuple of (nx, ny, nz)")
        nx, ny, nz = ndir
        cmd_parts.append(f"ndir {nx} {ny} {nz}")

    # Handle GPU option
    if gpu is not None:
        if isinstance(gpu, bool):
            cmd_parts.append(f"gpu {'on' if gpu else 'off'}")
        elif isinstance(gpu, str):
            cmd_parts.append(f"gpu {gpu}")

    # Handle GPUID option
    if gpuid is not None:
        cmd_parts.append(f"gpuid {gpuid}")

    # Handle DLB option
    if dlb is not None:
        cmd_parts.append(f"dlb {'on' if dlb else 'off'}")

    # Handle SPLIT option
    if split is not None:
        cmd_parts.append(f"split {'on' if split else 'off'}")

    # Handle PPANG option
    if ppang is not None:
        cmd_parts.append(f"ppang {ppang}")

    # Handle precision options
    if single:
        cmd_parts.append("single")
    elif double:
        cmd_parts.append("double")

    # Add any additional kwargs
    for key, value in kwargs.items():
        if isinstance(value, bool):
            cmd_parts.append(f"{key} {'on' if value else 'off'}")
        else:
            cmd_parts.append(f"{key} {value}")

    cmd = " ".join(cmd_parts)
    charmm_script(cmd)

    if show:
        import pycharmm.energy as eng
        eng.show()


def get_supported_restraints():
    """Get information about restraint support in DOMDEC.

    Returns
    -------
    dict
        Dictionary with 'supported' and 'unsupported' lists of restraint types.

    Examples
    --------
    >>> import pycharmm.domdec as domdec
    >>> info = domdec.get_supported_restraints()
    >>> print("Supported:", info['supported'])
    """
    return {
        'supported': [
            'DMCO - Distance matrix constraints',
            'RXNC - Umbrella potential',
            'ADUMB - Adaptive umbrella sampling',
            'RESD - Restrained distances',
            'NOE - Imposed distance restraints',
            'CONS HARM ABSO - Absolute harmonic constraints',
            'CONS DIHE - Dihedral constraints',
            'CONS HMCM - Center of mass constraints'
        ],
        'unsupported': [],
        'notes': {
            'general': 'DOMDEC supports most standard CHARMM restraint types.',
            'limitations': (
                'Only supports orthogonal simulation boxes. '
                'Only supports 3-atom solvent models (TIP3, SPC). '
                'Minimization does not work with DOMDEC.'
            ),
            'dlb': (
                'Dynamic Load Balancing is in beta phase for Constant Pressure simulations. '
                'If you have trouble, switch off DLB for CPT.'
            )
        }
    }
