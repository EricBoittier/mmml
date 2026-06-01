"""Partial ML / MM coupling via MLpot (stubs for future work)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.setup import (
    MlpotContext,
    register_mlpot,
)


@dataclass
class PartialMlMmConfig:
    """Configuration for a mixed ML region + MM environment."""

    ml_seg_id: str
    ml_charge: float = 0.0
    ml_fq: bool = True
    mlmm_ctonnb: Optional[float] = None
    mlmm_ctofnb: Optional[float] = None
    # When True, require ML–MM pair electrostatics in PyCharmm_Calculator (not implemented).
    use_mlmm_pair_lists: bool = False


def register_mlpot_partial_mm(
    pyCModel: Any,
    ml_Z: Sequence[int],
    config: PartialMlMmConfig,
) -> MlpotContext:
    """Register MLpot on a segment subset; MM atoms keep CHARMM nonbonded terms.

    Notes
    -----
    - ``get_pycharmm_calculator`` must honor ``ml_atom_indices`` from MLpot (see
      ``helper_mlp.get_pyc``).
    - ``PyCharmm_Calculator.calculate_charmm`` must consume ``idxu``/``idxv`` ML–MM
      pair lists when ``config.use_mlmm_pair_lists`` is True.

    Raises
    ------
    NotImplementedError
        If ``use_mlmm_pair_lists`` is True (electrostatic embedding stub).
    """
    if config.use_mlmm_pair_lists:
        raise NotImplementedError(
            "ML–MM pair electrostatics via idxu/idxv are not implemented in "
            "PyCharmm_Calculator yet. Set use_mlmm_pair_lists=False for segment-only "
            "registration (ML region + MM nonbonds)."
        )

    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_seg_id

    ml_sel = select_by_seg_id(config.ml_seg_id)
    return register_mlpot(
        pyCModel,
        ml_Z,
        ml_sel,
        ml_charge=config.ml_charge,
        ml_fq=config.ml_fq,
        mlmm_ctonnb=config.mlmm_ctonnb,
        mlmm_ctofnb=config.mlmm_ctofnb,
    )


def validate_partial_ml_indices(
    ml_atom_indices: Sequence[int],
    n_atoms: int,
) -> None:
    """Sanity-check ML atom indices against the loaded PSF."""
    idx = np.asarray(ml_atom_indices, dtype=int)
    if idx.size == 0:
        raise ValueError("ML atom selection is empty")
    if idx.min() < 0 or idx.max() >= n_atoms:
        raise ValueError(
            f"ML indices out of range [0, {n_atoms - 1}]: min={idx.min()}, max={idx.max()}"
        )
