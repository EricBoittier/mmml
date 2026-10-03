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

"""Build a crystal with any space group symmetry,
   optimise its lattice parameters and
   molecular coordinates and
   carry out a vibrational analysis using the options.

Corresponds to CHARMM command `CRYStal`

See CHARMM documentation [crystl](<https://academiccharmm.org/documentation/version/c47b1/crystl>)
for more information

Functions
=========
- `define_cubic` -- defines a cubic lattice for a new crystal
- `define_tetra` -- defines a tetragonal lattice for a new crystal
- `define_ortho` -- defines a orthorhombic lattice for a new crystal
- `define_mono` -- defines a monoclinic lattice for a new crystal
- `define_tri` -- defines a triclinic lattice for a new crystal
- `define_hexa` -- defines a hexagonal lattice for a new crystal
- `define_rhombo` -- defines a rhombohedral lattice for a new crystal
- `define_octa` -- defines a octahedral lattice for a new crystal
- `define_rhdo` -- defines a rhombic dodecahedron lattice for a new crystal
- `free`      -- turns off and frees all crystal and image operations
- `get_xucell` -- returns list of crystal lattice parameters [a ,b, c,alpha, beta, gamma]
- `get_xtltyp` -- returns string of crystal lattice type

Examples
========
Build a cubic simulation box of 0.7 nm
>>> crystal.define_cubic(70)
>>> crystal.build(12)
"""

import ctypes
import logging
from pycharmm.loader import lib
import pycharmm

# Module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Backend Compatibility
# =============================================================================

# Crystal type support by compute backend and features
# - blade: NVE/NVT dynamics support on BLaDE GPU
# - blade_npt: NPT (pressure coupling) support on BLaDE
# - openmm: OpenMM interface support
# - pme: PME/EWALD electrostatics support (requires orthorhombic)
# - domdec: Domain decomposition support in current DOMDEC engine
#
# CRITICAL: PME/EWALD only supports orthorhombic boxes.
# Non-orthorhombic crystals will cause CHARMM to die with:
#   "The periodic box is non orthorhombic (EWALD wont work)"
# DOMDEC compatibility is checked only when the DOMDEC backend is active.
#
CRYSTAL_BACKEND_SUPPORT = {
    'CUBI': {
        'blade': True, 'blade_npt': True, 'openmm': True,
        'pme': True, 'domdec': True, 'orthorhombic': True
    },
    'TETR': {
        'blade': True, 'blade_npt': 'restricted', 'openmm': False,
        'pme': True, 'domdec': True, 'orthorhombic': True
    },
    'ORTH': {
        'blade': True, 'blade_npt': True, 'openmm': True,
        'pme': True, 'domdec': True, 'orthorhombic': True
    },
    'RECT': {
        'blade': True, 'blade_npt': True, 'openmm': True,
        'pme': True, 'domdec': True, 'orthorhombic': True
    },
    'MONO': {
        'blade': True, 'blade_npt': False, 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
    'TRIC': {
        'blade': True, 'blade_npt': False, 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
    'HEXA': {
        'blade': True, 'blade_npt': 'restricted', 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
    'RHOM': {
        'blade': True, 'blade_npt': False, 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
    'OCTA': {
        'blade': True, 'blade_npt': True, 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
    'RHDO': {
        'blade': True, 'blade_npt': True, 'openmm': False,
        'pme': False, 'domdec': False, 'orthorhombic': False
    },
}


def _get_current_backend():
    """Get the current compute backend.

    Returns
    -------
    str
        'blade', 'openmm', 'domdec', or 'standard'
    """
    try:
        import pycharmm.blade as blade
        if blade.is_enabled():
            return 'blade'
    except (ImportError, AttributeError):
        pass

    # OpenMM detection would go here if there was a pycharmm.openmm module
    # For now, check via CHARMM parameter or command output
    try:
        # Check if OpenMM is active via param store or other mechanism
        pass
    except Exception:
        pass

    try:
        import pycharmm.domdec as domdec
        if domdec.is_enabled():
            return 'domdec'
    except (ImportError, AttributeError):
        pass

    return 'standard'


def _is_pme_enabled():
    """Check if PME/EWALD electrostatics are enabled.

    Returns
    -------
    bool
        True if PME is enabled, False otherwise
    """
    try:
        from pycharmm.loader import lib
        if hasattr(lib, 'nbondsdata_get_qewald'):
            lib.nbondsdata_get_qewald.restype = ctypes.c_int
            return bool(lib.nbondsdata_get_qewald())
    except (AttributeError, ImportError):
        pass

    # Fallback: assume PME might be used (safer to warn)
    return None  # Unknown


def _check_backend_compatibility(crystal_type: str) -> None:
    """Check and warn about backend compatibility issues.

    Parameters
    ----------
    crystal_type : str
        Crystal type code (e.g., 'CUBI', 'MONO', 'TRIC')

    Raises
    ------
    No exceptions raised, but logs warnings for incompatibilities.
    """
    backend = _get_current_backend()
    support = CRYSTAL_BACKEND_SUPPORT.get(crystal_type, {})

    # Check PME/EWALD compatibility (CRITICAL - will cause CHARMM to die)
    if not support.get('pme', True):
        pme_status = _is_pme_enabled()
        if pme_status is True:
            logger.error(
                f"Crystal type {crystal_type} is NOT compatible with PME/EWALD! "
                "CHARMM will die with 'non orthorhombic (EWALD wont work)'. "
                "Use cutoff-based electrostatics or switch to orthorhombic box."
            )
        elif pme_status is None:
            # Unknown PME status - warn anyway
            logger.warning(
                f"Crystal type {crystal_type} is NOT compatible with PME/EWALD. "
                "If you plan to use EWALD or PME electrostatics, switch to "
                "an orthorhombic box type (CUBI, TETR, ORTH, RECT)."
            )

    # Check DOMDEC compatibility only when DOMDEC is actually active.
    if backend == 'domdec' and not support.get('domdec', True):
        logger.warning(
            f"Crystal type {crystal_type}: Domain decomposition (domdec) NOT supported. "
            "Parallel simulations with domdec require orthorhombic boxes."
        )

    # Check BLaDE compatibility
    if backend == 'blade':
        if not support.get('blade', True):
            logger.warning(
                f"Crystal type {crystal_type} may not be fully supported by BLaDE"
            )
        if support.get('blade_npt') is False:
            logger.warning(
                f"Crystal type {crystal_type}: NPT dynamics NOT supported on BLaDE. "
                "Use NVE or NVT ensembles only."
            )
        elif support.get('blade_npt') == 'restricted':
            logger.info(
                f"Crystal type {crystal_type}: NPT dynamics has restricted support on BLaDE"
            )

    # Check OpenMM compatibility
    if backend == 'openmm' and not support.get('openmm', False):
        logger.warning(
            f"Crystal type {crystal_type} is NOT supported by OpenMM! "
            "OpenMM will force orthorhombic angles (alpha=beta=gamma=90)."
        )


def is_orthorhombic(crystal_type: str = None) -> bool:
    """Check if a crystal type has orthorhombic geometry (all angles = 90°).

    Orthorhombic boxes are required for:
    - PME/EWALD electrostatics
    - Domain decomposition (domdec)
    - OpenMM interface

    Parameters
    ----------
    crystal_type : str, optional
        Crystal type code. If None, queries current crystal.

    Returns
    -------
    bool
        True if orthorhombic, False otherwise
    """
    if crystal_type is None:
        crystal_type = get_crystal_type_direct()
        if crystal_type is None:
            # Fallback to heuristic
            crystal_type = get_xtltyp()

    support = CRYSTAL_BACKEND_SUPPORT.get(crystal_type, {})
    return support.get('orthorhombic', False)


# =============================================================================
# ctypes setup (ensures proper return types and parameter types)
# =============================================================================

_ctypes_initialized = False


def _init_ctypes():
    """Initialize ctypes declarations for crystal API functions."""
    global _ctypes_initialized
    if _ctypes_initialized:
        return

    try:


        # All crystal_define_* functions return c_int (1 for success)
        # Parameters are passed by reference (pointer to c_double)
        lib.crystal_define_cubic.restype = ctypes.c_int
        lib.crystal_define_cubic.argtypes = [ctypes.POINTER(ctypes.c_double)]

        lib.crystal_define_tetra.restype = ctypes.c_int
        lib.crystal_define_tetra.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_ortho.restype = ctypes.c_int
        lib.crystal_define_ortho.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_mono.restype = ctypes.c_int
        lib.crystal_define_mono.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_tri.restype = ctypes.c_int
        lib.crystal_define_tri.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_hexa.restype = ctypes.c_int
        lib.crystal_define_hexa.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_rhombo.restype = ctypes.c_int
        lib.crystal_define_rhombo.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]

        lib.crystal_define_octa.restype = ctypes.c_int
        lib.crystal_define_octa.argtypes = [ctypes.POINTER(ctypes.c_double)]

        lib.crystal_define_rhdo.restype = ctypes.c_int
        lib.crystal_define_rhdo.argtypes = [ctypes.POINTER(ctypes.c_double)]

        # crystal_build has complex variable-length array argument,
        # so we only set restype and let ctypes handle the args flexibly
        lib.crystal_build.restype = ctypes.c_int

        _ctypes_initialized = True
    except AttributeError as e:
        logger.debug(f"Could not initialize crystal ctypes: {e}")


def crystal_free_available() -> bool:
    """True when ``libcharmm`` exports ``crystal_free`` (KEY_LIBRARY rebuild)."""
    return callable(getattr(lib, "crystal_free", None))


def free_crystal() -> bool:
    """Clear crystal and periodic image state (``CRYSTAL FREE``)."""
    fn = getattr(lib, "crystal_free", None)
    if not callable(fn):
        return False
    return bool(fn())


def free():
    """Turns off and frees the data structures for crystal and image.

    KEY_LIBRARY builds have no ``crystal`` script command, so this prefers
    ``crystal_free`` when that export is linked.
    """
    if crystal_free_available():
        return free_crystal()
    free_script = pycharmm.script.CommandScript('crystal', free=True)
    free_script.run()
    return


def get_unit_cell() -> list[float]:
    """Unit cell edge lengths and angles (Å, degrees) from ``image_get_ucell``."""
    import pycharmm.image as image

    return image.get_ucell()


def get_cubic_side() -> float:
    """Cubic box edge length (Å) from ``XUCELL`` (assumes ``CUBI``)."""
    return float(get_unit_cell()[0])


def set_cubic_side(length: float, *, cutoff: float | None = None) -> bool:
    """Define a cubic crystal of ``length`` Å via ``crystal_define_cubic``.

    For full PBC setup (build + IMAGE), use
    ``mmml...pbc_env.prepare_charmm_pbc`` instead.
    """
    if not define_cubic(length):
        return False
    if cutoff is None:
        cutoff = min(18.0, max(float(length) / 2.0 - 2.0, 6.0))
    return bool(build(float(cutoff)))

def get_xtltyp():
    """Returns the string corresponding to the crystal lattice type"""
    from pycharmm import image
    xtl = image.get_ucell()
    if sum(xtl[3:5])/len(xtl[3:5]) == xtl[5] and xtl[5] == 90.0:
        if sum(xtl[:2])/len(xtl[:2]) == xtl[0]: xtltyp = 'CUBI'
        elif sum(xtl[:1])/len(xtl[:1]) == xtl[0]: xtltyp = 'TETR'
        elif sum(xtl[:1])/len(xtl[:1]) == xtl[0]: xtltyp = 'ORTH'
    elif (xtl[3]+xtl[5])/2 == 90.0 and xtl[4] != 90.0: xtltyp = 'MONO'
    elif xtl[0] == xtl[1] and sum(xtl[3:4])/2 == 90.0 and xtl[5] == 120.0: xtltyp = 'HEXA'
    elif sum(xtl[:2])/len(xtl[:2]) == xtl[0] and sum(xtl[3:5])/len(xtl[3:5]) == xtl[5] \
         and xtl[5] < 120.0: xtltyp = 'RHOM'
    elif sum(xtl[:2])/len(xtl[:2]) == xtl[0] and sum(xtl[3:5])/len(xtl[3:5]) == xtl[5] \
         and xtl[5]-109.471 < 1e-2: xtltyp = 'OCTA'
    elif sum(xtl[:2])/len(xtl[:2]) == xtl[0] and xtl[4] == xtl[5] and xtl[5] == 120.0 \
         and xtl[3] == 90.0: xtltyp = 'RHDO'
    else: xtltyp = 'TRIC'
    return xtltyp

def get_xtlacc(a=None, b=None, c=None, alpha=None, beta=None, gamma=None):
    """Returns the standard values of the three lattice vectors, a, b, c, in which
    a points along the x-axis
    
    Parameters (None)
    
    Output: 3x3 numpy array of lattice vectors
    
            [[a_x, a_y, a_z],
             [b_x, b_y, b_z],
             [c_x, c_y, c_z]]

    """
    import numpy as np
    if a is None:
        xucell = np.array(pycharmm.image.get_ucell())
    else:
        xucell = np.array([a, b, c, alpha, beta, gamma])
    degrad = np.pi/180.0
    xtlacc = np.zeros((3,3))
    xtlacc[0,0] = xucell[0]
    xtlacc[1,0] = xucell[1]*np.cos(degrad*xucell[5])
    xtlacc[1,1] = xucell[1]*np.sin(degrad*xucell[5])
    xtlacc[2,0] = xucell[2]*np.cos(degrad*xucell[4])
    xtlacc[2,1] = xucell[2]*(np.cos(degrad*xucell[3])\
                             -np.cos(degrad*xucell[4])\
                             *np.cos(degrad*xucell[5]))/np.sin(degrad*xucell[5])
    xtlacc[2,2]= np.sqrt(xucell[2]*xucell[2]\
                         -xtlacc[2,0]*xtlacc[2,0]-xtlacc[2,1]*xtlacc[2,1])
    return xtlacc


def define_cubic(length):
    """Defines a cubic lattice and constants for a new crystal

    Parameters
    ----------
    length : float
        length of all sides

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('CUBI')
    length = ctypes.c_double(length)
    success = lib.crystal_define_cubic(ctypes.byref(length))
    return success


def define_tetra(length_a, length_c):
    """Defines a tetragonal lattice and constants for a new crystal

    The alpha, beta and gamma angles are all 90.0 degrees.
    The length of sides a and b are equal.

    Parameters
    ----------
    length_a : float
        length of sides a and b
    length_c : float
        length of side c

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('TETR')
    length_a = ctypes.c_double(length_a)
    length_c = ctypes.c_double(length_c)
    success = lib.crystal_define_tetra(ctypes.byref(length_a),
                                              ctypes.byref(length_c))
    return success


def define_ortho(length_a, length_b, length_c):
    """Defines a orthorhombic lattice and constants for a new crystal

    The alpha, beta and gamma angles are all 90.0 degrees.

    Parameters
    ----------
    length_a : float
        length of side a
    length_b : float
        length of side b
    length_c : float
        length of side c

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('ORTH')
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    success = lib.crystal_define_ortho(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c))
    return success


def define_mono(length_a, length_b, length_c, angle_beta):
    """Defines a monoclinic lattice and constants for a new crystal

    The alpha and gamma angles are both 90.0 degrees.

    Note: NPT dynamics is NOT supported for monoclinic crystals on BLaDE.

    Parameters
    ----------
    length_a : float
        length of side a
    length_b : float
        length of side b
    length_c : float
        length of side c
    angle_beta : float
        measure of angle beta in degrees

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('MONO')
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    angle_beta = ctypes.c_double(angle_beta)
    success = lib.crystal_define_mono(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c),
                                              ctypes.byref(angle_beta))
    return success


def define_tri(length_a, length_b, length_c,
               angle_alpha, angle_beta, angle_gamma):
    """Defines a triclinic lattice and constants for a new crystal

    Note: NPT dynamics is NOT supported for triclinic crystals on BLaDE.
    OpenMM does NOT support triclinic crystals (will force orthorhombic).

    Parameters
    ----------
    length_a : float
        length of side a
    length_b : float
        length of side b
    length_c : float
        length of side c
    angle_alpha : float
        measure of angle alpha in degrees
    angle_beta : float
        measure of angle beta in degrees
    angle_gamma : float
        measure of angle gamma in degrees

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('TRIC')
    length_a = ctypes.c_double(length_a)
    length_b = ctypes.c_double(length_b)
    length_c = ctypes.c_double(length_c)
    angle_alpha = ctypes.c_double(angle_alpha)
    angle_beta = ctypes.c_double(angle_beta)
    angle_gamma = ctypes.c_double(angle_gamma)
    success = lib.crystal_define_tri(ctypes.byref(length_a),
                                              ctypes.byref(length_b),
                                              ctypes.byref(length_c),
                                              ctypes.byref(angle_alpha),
                                              ctypes.byref(angle_beta),
                                              ctypes.byref(angle_gamma))
    return success


def define_hexa(length_a, length_c):
    """Defines a hexagonal lattice and constants for a new crystal

    Note: NPT dynamics has restricted support for hexagonal crystals on BLaDE.

    Parameters
    ----------
    length_a : float
        lengths of sides a and b
    length_c : float
        length of side c

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('HEXA')
    length_a = ctypes.c_double(length_a)
    length_c = ctypes.c_double(length_c)
    success = lib.crystal_define_hexa(ctypes.byref(length_a),
                                             ctypes.byref(length_c))
    return success


def define_rhombo(length, angle):
    """Defines a rhombohedral lattice and constants for a new crystal

    Note: NPT dynamics is NOT supported for rhombohedral crystals on BLaDE.

    Parameters
    ----------
    length : float
        length of each side
    angle : float
        measure of each angle in degrees, must be between 0 and 120

    Returns
    -------
    bool
        True for success, otherwise False
    """
    if angle <= 0.0 or angle >= 120.0:
        raise ValueError("Value %d out of range (0.0, 120.0)" % (angle,))

    _init_ctypes()
    _check_backend_compatibility('RHOM')
    length = ctypes.c_double(length)
    angle = ctypes.c_double(angle)
    success = lib.crystal_define_rhombo(ctypes.byref(length),
                                               ctypes.byref(angle))
    return success


def define_octa(length):
    """Defines a truncated octahedral lattice and constants for a new crystal

    Note: Not supported by OpenMM (will force orthorhombic).

    Parameters
    ----------
    length : float
        the length of each side

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('OCTA')
    length = ctypes.c_double(length)
    success = lib.crystal_define_octa(ctypes.byref(length))
    return success


def define_rhdo(length):
    """Defines a rhombic dodecahedron lattice and constants for a new crystal

    Note: Not supported by OpenMM (will force orthorhombic).

    Parameters
    ----------
    length : float
        the length of each side

    Returns
    -------
    bool
        True for success, otherwise False
    """
    _init_ctypes()
    _check_backend_compatibility('RHDO')
    length = ctypes.c_double(length)
    success = lib.crystal_define_rhdo(ctypes.byref(length))
    return success


def build(cutoff, sym_ops=None):
    """Build the crystal by repeatedly applying specified transformations

    Parameters
    ----------
    cutoff : float
        images within cutoff distance are included in transformation list
    sym_ops : list[str], optional
        Symmetry operations in crystallographic notation WITH parentheses.
        Examples: '(X,Y,Z)', '(-X,Y,-Z)', '(X+1/2,Y,Z)', '(-X,-Y,-Z)'

    Returns
    -------
    bool
        True for success, otherwise False

    Notes
    -----
    - Symmetry operations MUST include parentheses: '(X,Y,Z)' not 'X,Y,Z'
    - Identity (X,Y,Z) is automatically added as the first operation
    - Do not pass identity explicitly or you'll get a "duplicate operation" error
    - Pass non-identity operations like inversion '(-X,-Y,-Z)' or glide '(X+1/2,Y,Z)'

    Examples
    --------
    >>> crystal.define_cubic(50.0)
    >>> crystal.build(cutoff=12.0)  # No sym_ops, just identity
    >>> crystal.build(cutoff=12.0, sym_ops=['(-X,-Y,-Z)'])  # With inversion
    """
    _init_ctypes()
    if sym_ops is None:
        sym_ops = []

    nops = len(sym_ops)

    # Keep references to encoded strings to prevent garbage collection
    # (ctypes.c_char_p doesn't keep a reference to the Python bytes object)
    encoded_ops = [s.encode('utf-8') for s in sym_ops]

    # Create array of c_char_p (pointers to strings)
    # Need at least 1 element to avoid empty array issues with ctypes
    str_array = (ctypes.c_char_p * max(1, nops))()
    for i, encoded in enumerate(encoded_ops):
        str_array[i] = encoded

    cutoff_c = ctypes.c_double(cutoff)
    nops_c = ctypes.c_int(nops)

    # Pass array directly (not byref) - ctypes arrays are already passed by reference
    # nops is passed by value (Fortran has 'value' attribute)
    success = lib.crystal_build(ctypes.byref(cutoff_c),
                                           str_array,
                                           nops_c)
    return success


# =============================================================================
# Query Functions (Direct API access)
# =============================================================================

def is_defined():
    """Check if a crystal is currently defined.

    Returns
    -------
    bool
        True if crystal is defined, False otherwise
    """
    try:

        lib.crystaldata_is_defined.restype = ctypes.c_int
        return bool(lib.crystaldata_is_defined())
    except AttributeError:
        # Fallback: check via get_crystal_type
        return get_crystal_type_direct() is not None


def get_crystal_type_direct():
    """Get the current crystal type directly from CHARMM memory.

    This queries XTLTYP directly from CHARMM, unlike get_xtltyp()
    which infers the type from unit cell parameters.

    Returns
    -------
    str or None
        Crystal type code ('CUBI', 'ORTH', 'RECT', 'MONO', 'TRIC',
        'HEXA', 'RHOM', 'OCTA', 'RHDO', 'TETR') or None if not defined
    """
    try:

        lib.crystaldata_get_type.restype = ctypes.c_int
        lib.crystaldata_get_type.argtypes = [ctypes.c_char * 5]

        out_type = (ctypes.c_char * 5)()
        success = lib.crystaldata_get_type(out_type)

        if success:
            return out_type.value.decode('utf-8').strip()
        return None
    except AttributeError:
        logger.debug("crystaldata_get_type not available - API not compiled")
        return None


def get_symmetry_count():
    """Get the number of symmetry operations in the current crystal.

    Returns
    -------
    int or None
        Number of symmetry operations (XNSYMM), or None if not available
    """
    try:

        lib.crystaldata_get_nsymm.restype = ctypes.c_int
        return int(lib.crystaldata_get_nsymm())
    except AttributeError:
        logger.debug("crystaldata_get_nsymm not available")
        return None


def get_cutoff():
    """Get the crystal image cutoff distance.

    Returns
    -------
    float or None
        Cutoff distance (CUTXTL) in Angstroms, or None if not available
    """
    try:

        lib.crystaldata_get_cutoff.restype = ctypes.c_double
        return float(lib.crystaldata_get_cutoff())
    except AttributeError:
        logger.debug("crystaldata_get_cutoff not available")
        return None


def get_unit_cell():
    """Get current unit cell parameters.

    This is a convenience wrapper around pycharmm.image.get_ucell().

    Returns
    -------
    tuple
        (a, b, c, alpha, beta, gamma) - lengths in Angstroms, angles in degrees
    """
    from pycharmm import image
    ucell = image.get_ucell()
    return tuple(ucell)


def get_transformation_count():
    """Get the number of image transformations.

    This is a convenience wrapper around pycharmm.image.get_ntrans().

    Returns
    -------
    int
        Number of image transformations (NTRANS)
    """
    from pycharmm import image
    return image.get_ntrans()
