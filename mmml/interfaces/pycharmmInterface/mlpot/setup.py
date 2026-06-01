"""Register PhysNet (or other) models with ``pycharmm.MLpot``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def sync_charmm_positions(positions: np.ndarray) -> None:
    """Push ``(N, 3)`` positions into the CHARMM coordinate array."""
    import pycharmm.coor as coor

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {arr.shape}")
    n = coor.get_natom()
    if arr.shape[0] != n:
        raise ValueError(f"positions rows {arr.shape[0]} != CHARMM natom {n}")
    coor.set_positions(pd.DataFrame(arr, columns=["x", "y", "z"]))


def get_charmm_positions_array() -> np.ndarray:
    """Read CHARMM coordinates as ``(N, 3)``."""
    import pycharmm.coor as coor

    df = coor.get_positions()
    return df[["x", "y", "z"]].to_numpy(dtype=float)


@dataclass
class MlpotContext:
    """Active MLpot registration (call :meth:`unset` when finished)."""

    mlpot: Any
    pyCModel: Any
    params: Any
    model: Any

    def unset(self) -> None:
        self.mlpot.unset_mlpot()


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401 — CHARMM env
    import pycharmm

    return pycharmm


def select_all_atoms():
    """CHARMM selection of all atoms."""
    return _import_pycharmm().SelectAtoms().all_atoms()


def select_by_seg_id(seg_id: str):
    """CHARMM selection by segment ID (e.g. ``'AMM1'`` for an ML region)."""
    return _import_pycharmm().SelectAtoms(seg_id=seg_id)


def select_by_resid(resid: int | str):
    """CHARMM selection by residue ID (e.g. ``1`` for the first residue)."""
    return _import_pycharmm().SelectAtoms(res_id=str(resid))


def setup_default_nbonds(
    *,
    cutnb: float = 14.0,
    ctofnb: float = 12.0,
    ctonnb: float = 10.0,
) -> None:
    """Apply a standard atom-based nonbond setup (matches mlpot test scripts)."""
    pycharmm = _import_pycharmm()
    script = f"""
    nbonds atom cutnb {cutnb} ctofnb {ctofnb} ctonnb {ctonnb} -
    vswitch -
    inbfrq -1 imgfrq -1
    """
    pycharmm.lingo.charmm_script(script)


def load_physnet_mlpot_bundle(
    checkpoint: PathLike,
    n_atoms: int,
    ase_atoms: Any,
) -> tuple[Any, Any, Any]:
    """Load checkpoint and build the ``get_pyc`` model wrapper for MLpot.

    Returns ``(params, model, pyCModel)``.
    """
    from mmml.cli.base import load_physnet_params_and_ef_model

    ckpt = Path(checkpoint).expanduser().resolve()
    params, model = load_physnet_params_and_ef_model(ckpt, natoms=n_atoms)
    model.natoms = n_atoms

    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_pyc

    pyCModel = get_pyc(params, model, ase_atoms)
    return params, model, pyCModel


def register_mlpot(
    pyCModel: Any,
    ml_Z: Sequence[int],
    ml_selection: Any,
    *,
    ml_charge: float = 0,
    ml_fq: bool = True,
    mlmm_ctonnb: Optional[float] = None,
    mlmm_ctofnb: Optional[float] = None,
    **kwargs: Any,
) -> MlpotContext:
    """Register ``pycharmm.MLpot`` and return a context manager-like handle."""
    pycharmm = _import_pycharmm()
    mlpot = pycharmm.MLpot(
        ml_model=pyCModel,
        ml_Z=list(ml_Z),
        ml_selection=ml_selection,
        ml_charge=ml_charge,
        ml_fq=ml_fq,
        mlmm_ctonnb=mlmm_ctonnb,
        mlmm_ctofnb=mlmm_ctofnb,
        **kwargs,
    )
    return MlpotContext(mlpot=mlpot, pyCModel=pyCModel, params=None, model=None)
