"""Host-side CHARMM ctypes copies used every MLpot force callback."""

from __future__ import annotations

import numpy as np


def stack_charmm_xyz(x, y, z, n: int) -> np.ndarray:
    """``(n, 3)`` float64 positions from CHARMM ``x/y/z`` without a Python list."""
    n = int(n)
    try:
        return np.stack(
            [
                np.ctypeslib.as_array(x, shape=(n,))[:n],
                np.ctypeslib.as_array(y, shape=(n,))[:n],
                np.ctypeslib.as_array(z, shape=(n,))[:n],
            ],
            axis=1,
        ).astype(np.float64, copy=True)
    except (TypeError, ValueError):
        return np.array([x[:n], y[:n], z[:n]], dtype=np.float64).T


def subtract_forces_from_charmm_grad(dx, dy, dz, forces, n: int) -> None:
    """``dx[i] -= F[i,0]`` (and y/z) via numpy views instead of a Python loop."""
    f = np.asarray(forces, dtype=np.float64)
    n = int(n)
    for arr, col in ((dx, 0), (dy, 1), (dz, 2)):
        try:
            view = np.ctypeslib.as_array(arr, shape=(n,))
            view[:n] -= f[:n, col]
            continue
        except (TypeError, ValueError):
            pass
        try:
            arr[:n] -= f[:n, col]
        except Exception:
            for i in range(n):
                arr[i] -= float(f[i, col])
