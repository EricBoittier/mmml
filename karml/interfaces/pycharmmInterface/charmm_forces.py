"""Fast reader for CHARMM's force array.

``pycharmm.coor.get_forces()`` boxes ``3 * natom`` Python floats into list
comprehensions and assembles a DataFrame. That costs ~205 us for a 10-atom system --
an order of magnitude more than the ``ENER FORCE`` evaluation it follows (~20 us) --
and grows linearly with atom count. This module reads the same C buffers straight
into numpy, which is bitwise identical and ~50-150x cheaper.

Imports of ``pycharmm`` are deferred into the functions: importing this module must
not load libcharmm (see the note in ``mmml_calculator``).
"""

from __future__ import annotations

import ctypes

import numpy as np


def charmm_gradient_array() -> np.ndarray:
    """CHARMM's ``dx/dy/dz`` energy gradient (``dE/dx``) as an ``(natom, 3)`` array.

    Reflects the last ``ENER FORCE``; it does not evaluate the energy itself.
    """
    import pycharmm.coor as coor

    natom = int(coor.get_natom())
    if natom == 0:
        return np.zeros((0, 3), dtype=np.float64)

    try:
        from pycharmm import lib

        buf = ctypes.c_double * natom
        c_dx, c_dy, c_dz = buf(), buf(), buf()
        lib.charmm.coor_get_forces(c_dx, c_dy, c_dz)

        grad = np.empty((natom, 3), dtype=np.float64)
        grad[:, 0] = np.ctypeslib.as_array(c_dx)
        grad[:, 1] = np.ctypeslib.as_array(c_dy)
        grad[:, 2] = np.ctypeslib.as_array(c_dz)
        return grad
    except (ImportError, AttributeError):
        # Older/patched pycharmm without the raw symbol: fall back to the DataFrame.
        return coor.get_forces()[["dx", "dy", "dz"]].to_numpy(dtype=np.float64)


def charmm_forces_array() -> np.ndarray:
    """Physical per-atom forces (kcal/mol/Å) as an ``(natom, 3)`` array.

    The force is the negative energy gradient.
    """
    grad = charmm_gradient_array()
    np.negative(grad, out=grad)
    return grad


def charmm_positions_array() -> np.ndarray:
    """Current CHARMM coordinates (Å) as an ``(natom, 3)`` array."""
    import pycharmm.coor as coor

    natom = int(coor.get_natom())
    if natom == 0:
        return np.zeros((0, 3), dtype=np.float64)

    try:
        from pycharmm import lib

        buf = ctypes.c_double * natom
        c_x, c_y, c_z = buf(), buf(), buf()
        lib.charmm.coor_get_positions(c_x, c_y, c_z)

        pos = np.empty((natom, 3), dtype=np.float64)
        pos[:, 0] = np.ctypeslib.as_array(c_x)
        pos[:, 1] = np.ctypeslib.as_array(c_y)
        pos[:, 2] = np.ctypeslib.as_array(c_z)
        return pos
    except (ImportError, AttributeError):
        return coor.get_positions()[["x", "y", "z"]].to_numpy(dtype=np.float64)


def set_charmm_positions_array(positions) -> None:
    """Push an ``(natom, 3)`` array of coordinates (Å) into CHARMM.

    Avoids the DataFrame round-trip that ``coor.set_positions`` requires.
    """
    import pycharmm.coor as coor

    natom = int(coor.get_natom())
    xyz = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
    if xyz.shape != (natom, 3):
        raise ValueError(f"expected positions of shape ({natom}, 3), got {xyz.shape}")

    try:
        from pycharmm import lib

        buf = ctypes.c_double * natom
        c_x, c_y, c_z = buf(), buf(), buf()
        np.ctypeslib.as_array(c_x)[:] = xyz[:, 0]
        np.ctypeslib.as_array(c_y)[:] = xyz[:, 1]
        np.ctypeslib.as_array(c_z)[:] = xyz[:, 2]
        lib.charmm.coor_set_positions(c_x, c_y, c_z)
    except (ImportError, AttributeError):
        import pandas as pd

        coor.set_positions(pd.DataFrame(xyz, columns=["x", "y", "z"]))
