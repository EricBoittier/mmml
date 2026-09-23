"""Strain-virial correction for the MLpot USER term under CHARMM CPT.

CHARMM's ``VIRAL`` computes the virial as ``sum_i x_i F_i`` over the forces the
MLpot callback writes to the central atoms. Those forces come from
minimum-image distances, so for any interaction across the cell boundary the
sum misses the lattice term and is not ``-dE/d(strain)``: the pressure CPT
couples to is wrong (32 Å ACO/DCM PhysNet students: 0.7-1.7 katm too high from
the ML term alone).

With positions and cell scaled together, ``x' = (1 + e) x`` and
``h' = (1 + e) h``,

    -dE/de_ab = sum_i F_ia x_ib  -  sum_c (dE/dh)_ac h_bc .

The first term is what ``VIRAL`` already has (in its ``x_a F_b`` order); the
second needs ``dE/dh`` at fixed positions, which the jitted forward returns
when the energy is differentiable in the cell (lattice shifts written as
``-stop_gradient(n) @ h``). The callback also evaluates on a rewrapped copy of
the coordinates, so ``sum_i (x_copy - x_charmm)_i F_i`` is added to make the
total equal the strain virial at CHARMM's coordinates. The correction goes to
CHARMM through ``mlpot_set_virial`` (``api_func.F90``); ``ENERGY`` adds it to
``EPRESS(VIXX:VIZZ)`` / ``EPROP(VIRI)`` right after ``VIRAL``.

It is computed only while a CPT segment with a live barostat is running
(:func:`cpt_strain_virial_scope`, entered by ``run_dynamics``), so NVE/NVT pay
nothing. ``MMML_MLPOT_STRAIN_VIRIAL=1`` forces it on, ``0`` off.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

import numpy as np

STRAIN_VIRIAL_ENV = "MMML_MLPOT_STRAIN_VIRIAL"

_CPT_ACTIVE = False


def strain_virial_enabled() -> bool:
    """True when the callback must compute and stage the virial correction."""
    raw = (os.environ.get(STRAIN_VIRIAL_ENV) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return _CPT_ACTIVE


def _barostat_live(dynamics_kwargs: dict[str, Any]) -> bool:
    if not bool(dynamics_kwargs.get("cpt")):
        return False
    pmass = dynamics_kwargs.get("pmass")
    try:
        return pmass is not None and float(pmass) > 0.0
    except (TypeError, ValueError):
        return False


@contextlib.contextmanager
def cpt_strain_virial_scope(dynamics_kwargs: dict[str, Any]) -> Iterator[bool]:
    """Enable the correction for the duration of a CPT ``DYNA`` with ``pmass > 0``."""
    global _CPT_ACTIVE
    previous = _CPT_ACTIVE
    active = _barostat_live(dynamics_kwargs)
    if active:
        require_charmm_virial_hook()
        print(
            f"MLpot strain virial: CPT barostat live (pmass={dynamics_kwargs.get('pmass')}); "
            "the callback stages -dE/d(strain) corrections for CHARMM's pressure",
            flush=True,
        )
    _CPT_ACTIVE = previous or active
    try:
        yield active
    finally:
        _CPT_ACTIVE = previous


def _charmm_setter():
    try:
        import pycharmm.lib as lib
    except (ImportError, OSError):
        return None
    return getattr(lib.charmm, "mlpot_set_virial", None)


def require_charmm_virial_hook() -> None:
    """Fail closed: NpT with a libcharmm that cannot take the correction is wrong physics."""
    if _charmm_setter() is None:
        raise RuntimeError(
            "CPT NpT with MLpot needs libcharmm built with mlpot_set_virial "
            "(setup/charmm/source/api/api_func.F90); rebuild with "
            "scripts/rebuild_charmm_mlpot.sh. Without it CHARMM's pressure uses the "
            "central-atom sum(x*F) of the ML forces, which is not the strain virial."
        )


def virial_correction_kcal(
    *,
    dE_dcell_eV: Any,
    cell: Any,
    forces_eV_A: Any,
    positions_eval: Any,
    positions_charmm: Any,
    ev_to_kcal: float,
) -> np.ndarray:
    """``(3, 3)`` correction in CHARMM ``VIRAL`` order (element ``[a, b]`` ~ ``sum x_a F_b``), kcal/mol."""
    G = np.asarray(dE_dcell_eV, dtype=np.float64).reshape(3, 3)
    h = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    F = np.asarray(forces_eV_A, dtype=np.float64).reshape(-1, 3)
    dx = np.asarray(positions_eval, dtype=np.float64).reshape(-1, 3) - np.asarray(
        positions_charmm, dtype=np.float64
    ).reshape(-1, 3)
    # (-dE/de)_ab - sum_i F_ia x_ib = -(G h^T)_ab ; VIRAL element [b, a] holds sum x_b F_a.
    lattice = (-(G @ h.T)).T
    rewrap = dx.T @ F  # [a, b] = sum_i dx_ia F_ib, already in VIRAL order
    return (lattice + rewrap) * float(ev_to_kcal)


def push_virial_to_charmm(correction_kcal: np.ndarray) -> None:
    """Stage the correction for this ``ENERGY`` call (consumed after ``VIRAL``)."""
    import ctypes

    setter = _charmm_setter()
    if setter is None:
        require_charmm_virial_hook()
        return
    arr = (ctypes.c_double * 9)(*np.asarray(correction_kcal, dtype=np.float64).reshape(9))
    setter(arr)
