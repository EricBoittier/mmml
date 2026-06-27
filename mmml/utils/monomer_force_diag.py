"""Per-monomer hybrid force diagnostics for selective geometry recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import forces_grms_kcalmol_A


@dataclass(frozen=True)
class MonomerForceDiag:
    """Per-monomer GRMS from hybrid forces and indices flagged for patching."""

    grms_per_monomer: np.ndarray
    flagged: tuple[int, ...]
    cluster_grms: float
    threshold: float

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    def use_selective_repack(self, *, max_select: int = 2) -> bool:
        return 0 < self.n_flagged <= int(max_select)


def per_monomer_forces_grms_kcalmol_A(
    forces_kcal: np.ndarray,
    monomer_offsets: np.ndarray,
) -> np.ndarray:
    """RMS force per monomer (kcal/mol/Å), matching cluster GRMS convention."""
    f = np.asarray(forces_kcal, dtype=np.float64).reshape(-1, 3)
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_monomers = int(len(offsets) - 1)
    grms = np.empty(n_monomers, dtype=np.float64)
    for mi in range(n_monomers):
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        grms[mi] = forces_grms_kcalmol_A(f[s:e])
    return grms


def select_stressed_monomer_indices(
    grms_per_monomer: np.ndarray,
    *,
    max_select: int = 2,
    min_abs_grms: float = 30.0,
    min_ratio_to_median: float = 2.5,
) -> list[int]:
    """Return monomer indices with clearly elevated per-monomer GRMS.

    Selective repack is appropriate when only one or two monomers dominate
    the cluster stress; widespread elevation returns an empty list so callers
    can fall back to global repack.
    """
    g = np.asarray(grms_per_monomer, dtype=np.float64).reshape(-1)
    if g.size <= 1:
        return []
    median = float(np.median(g))
    candidates = [
        int(i)
        for i, value in enumerate(g)
        if float(value) >= float(min_abs_grms)
        and (median < 1.0e-6 or float(value) >= float(min_ratio_to_median) * median)
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda i: g[i], reverse=True)
    top = candidates[: max(1, int(max_select))]
    if len(top) > int(max_select):
        return []
    rest = [float(g[i]) for i in range(g.size) if i not in top]
    if rest:
        top_mean = float(np.mean([g[i] for i in top]))
        rest_mean = float(np.mean(rest))
        if rest_mean > 1.0e-6 and top_mean < 1.35 * rest_mean:
            return []
    return top


def diagnose_monomer_forces(
    forces_kcal: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    max_select: int = 2,
    min_abs_grms: float = 30.0,
    min_ratio_to_median: float = 2.5,
    overlap_monomers: tuple[int, ...] = (),
) -> MonomerForceDiag:
    """Build per-monomer force diagnostics and merge overlap hints."""
    grms = per_monomer_forces_grms_kcalmol_A(forces_kcal, monomer_offsets)
    flagged = select_stressed_monomer_indices(
        grms,
        max_select=max_select,
        min_abs_grms=min_abs_grms,
        min_ratio_to_median=min_ratio_to_median,
    )
    if overlap_monomers:
        merged = sorted({int(i) for i in flagged} | {int(i) for i in overlap_monomers})
        if len(merged) <= int(max_select):
            flagged = merged
        else:
            flagged = []
    cluster = forces_grms_kcalmol_A(forces_kcal)
    threshold = float(min_abs_grms)
    if grms.size:
        med = float(np.median(grms))
        if med > 1.0e-6:
            threshold = max(threshold, float(min_ratio_to_median) * med)
    return MonomerForceDiag(
        grms_per_monomer=grms,
        flagged=tuple(int(i) for i in flagged),
        cluster_grms=float(cluster),
        threshold=float(threshold),
    )


def mlpot_hybrid_forces_kcalmol_A(
    mlpot_ctx: Any,
    *,
    positions: np.ndarray | None = None,
    natom: int | None = None,
) -> np.ndarray | None:
    """Hybrid ML/MM forces (kcal/mol/Å) at CHARMM or given positions."""
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_positions_angstrom,
        mlpot_spherical_forces_ev_angstrom,
    )
    from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if pyCModel is None:
        return None

    if natom is None:
        import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
        import pycharmm.coor as coor

        natom = int(coor.get_natom())
    n = int(natom)

    if positions is None:
        pos = charmm_positions_angstrom()[:n]
    else:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[:n]

    use_pbc = bool(getattr(mlpot_ctx, "use_pbc", False))
    box_A = getattr(mlpot_ctx, "cubic_box_side_A", None)
    if box_A is None:
        box_A = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)

    forces_ev = mlpot_spherical_forces_ev_angstrom(
        pyCModel,
        positions=pos,
        use_pbc=use_pbc,
        box_A=float(box_A) if box_A is not None else None,
    )
    if forces_ev is not None and int(forces_ev.shape[0]) >= n:
        return np.asarray(forces_ev[:n], dtype=np.float64) * float(ev2kcalmol)

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        mlpot_last_hybrid_forces_kcalmol_A,
    )

    forces_kcal = mlpot_last_hybrid_forces_kcalmol_A(pyCModel)
    if forces_kcal is not None and int(forces_kcal.shape[0]) >= n:
        return np.asarray(forces_kcal[:n], dtype=np.float64)
    return None


def resolve_selective_repack_monomers(
    mlpot_ctx: Any,
    monomer_offsets: np.ndarray,
    *,
    max_select: int = 2,
    min_abs_grms: float = 30.0,
    min_ratio_to_median: float = 2.5,
    overlap_monomers: tuple[int, ...] = (),
    positions: np.ndarray | None = None,
) -> MonomerForceDiag | None:
    """Diagnose hybrid forces and return selective-repack candidates, if any."""
    forces = mlpot_hybrid_forces_kcalmol_A(mlpot_ctx, positions=positions)
    if forces is None:
        return None
    offsets = np.asarray(monomer_offsets, dtype=int)
    n_atoms = int(offsets[-1]) if offsets.size else 0
    if int(forces.shape[0]) < n_atoms:
        return None
    diag = diagnose_monomer_forces(
        forces[:n_atoms],
        offsets,
        max_select=max_select,
        min_abs_grms=min_abs_grms,
        min_ratio_to_median=min_ratio_to_median,
        overlap_monomers=overlap_monomers,
    )
    if not diag.use_selective_repack(max_select=max_select):
        return None
    return diag
