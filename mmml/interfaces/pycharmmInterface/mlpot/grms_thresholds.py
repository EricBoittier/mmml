"""Size- and monomer-aware GRMS thresholds for liquid-prep gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    charmm_total_forces_kcalmol_A,
    forces_grms_kcalmol_A,
    mlpot_hybrid_grms_from_calculator,
)


@dataclass(frozen=True)
class MonomerGrmsStats:
    """Per-monomer and total GRMS from CHARMM and optional hybrid forces."""

    charmm_per_monomer: np.ndarray
    hybrid_per_monomer: np.ndarray | None
    charmm_total: float
    hybrid_total: float | None

    @property
    def n_monomers(self) -> int:
        return int(self.charmm_per_monomer.size)


@dataclass(frozen=True)
class GrmsThresholds:
    """Early-intervention and pre-dynamics GRMS ceilings (kcal/mol/Å)."""

    intervention_grms: float
    max_grms_before_dyn: float
    charmm_p90: float
    hybrid_p90: float | None
    notes: str


def per_monomer_grms_from_forces(
    forces: np.ndarray,
    atoms_per_list: list[int] | tuple[int, ...],
) -> np.ndarray:
    """RMS force magnitude per monomer (kcal/mol/Å convention of input forces)."""
    f = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    counts = [int(x) for x in atoms_per_list]
    if int(np.sum(counts)) != int(f.shape[0]):
        raise ValueError(
            f"force rows ({f.shape[0]}) != sum(atoms_per_list) ({int(np.sum(counts))})"
        )
    out: list[float] = []
    start = 0
    for n in counts:
        block = f[start : start + n]
        out.append(forces_grms_kcalmol_A(block))
        start += n
    return np.asarray(out, dtype=np.float64)


def measure_monomer_grms_stats(
    atoms_per_list: list[int] | tuple[int, ...],
    *,
    mlpot_ctx: Any | None = None,
    hybrid_forces_kcal: np.ndarray | None = None,
) -> MonomerGrmsStats:
    """Measure CHARMM and optional hybrid per-monomer GRMS at current coordinates."""
    charmm_f = charmm_total_forces_kcalmol_A()
    charmm_per = per_monomer_grms_from_forces(charmm_f, atoms_per_list)
    charmm_total = forces_grms_kcalmol_A(charmm_f)

    hybrid_per: np.ndarray | None = None
    hybrid_total: float | None = None
    if hybrid_forces_kcal is not None:
        hybrid_per = per_monomer_grms_from_forces(hybrid_forces_kcal, atoms_per_list)
        hybrid_total = forces_grms_kcalmol_A(hybrid_forces_kcal)
    elif mlpot_ctx is not None:
        hybrid_total = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
        if hybrid_total is not None and np.isfinite(hybrid_total):
            from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
                charmm_positions_angstrom,
                mlpot_spherical_forces_ev_angstrom,
            )
            from mmml.interfaces.pycharmmInterface.mmml_calculator import ev2kcalmol

            pyCModel = getattr(mlpot_ctx, "pyCModel", None)
            if pyCModel is not None:
                pos = charmm_positions_angstrom()
                box = getattr(mlpot_ctx, "cubic_box_side_A", None)
                if box is None:
                    box = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)
                forces_ev = mlpot_spherical_forces_ev_angstrom(
                    pyCModel,
                    positions=pos,
                    use_pbc=bool(getattr(mlpot_ctx, "use_pbc", False)),
                    box_A=float(box) if box is not None else None,
                )
                if forces_ev is not None:
                    hybrid_f = np.asarray(forces_ev, dtype=np.float64) * float(ev2kcalmol)
                    hybrid_per = per_monomer_grms_from_forces(hybrid_f, atoms_per_list)
                    hybrid_total = forces_grms_kcalmol_A(hybrid_f)
                    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
                        mlpot_skip_charmm_ener_force_before_first_sd,
                    )

                    if mlpot_skip_charmm_ener_force_before_first_sd(mlpot_ctx):
                        from mmml.interfaces.pycharmmInterface.charmm_mpi import (
                            recover_mpi_for_charmm_after_jax,
                        )

                        recover_mpi_for_charmm_after_jax(
                            phase="after pre-SD hybrid GRMS JAX eval",
                        )

    return MonomerGrmsStats(
        charmm_per_monomer=charmm_per,
        hybrid_per_monomer=hybrid_per,
        charmm_total=float(charmm_total),
        hybrid_total=float(hybrid_total) if hybrid_total is not None else None,
    )


# Ignore stale CHARMM force-buffer slots when deriving thresholds (pre-SD may skip
# ENER FORCE under deferred JAX / MPI).
_MONOMER_GRMS_CEILING_KCALMOL_A = 1000.0


def _robust_monomer_grms(values: np.ndarray, *, default: float) -> np.ndarray:
    """Finite per-monomer GRMS samples within a physical ceiling."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= 0.0) & (arr <= _MONOMER_GRMS_CEILING_KCALMOL_A)]
    if arr.size == 0:
        return np.array([float(default)], dtype=np.float64)
    return arr


def _percentile_safe(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def resolve_grms_thresholds_from_stats(
    stats: MonomerGrmsStats,
    *,
    n_monomers: int,
    n_atoms: int,
    pbc: bool = False,
    base_max_grms: float = 50.0,
    charmm_bonded_ok_max: float = 5.0,
) -> GrmsThresholds:
    """Derive intervention and dynamics GRMS limits from per-monomer MM / hybrid stats."""
    n_mol = max(1, int(n_monomers))
    n_at = max(1, int(n_atoms))

    charmm_samples = _robust_monomer_grms(
        stats.charmm_per_monomer, default=charmm_bonded_ok_max
    )
    charmm_p90 = _percentile_safe(charmm_samples, 90.0, charmm_bonded_ok_max)
    charmm_max = float(np.max(charmm_samples))

    hybrid_p90: float | None = None
    hybrid_max: float | None = None
    if stats.hybrid_per_monomer is not None and stats.hybrid_per_monomer.size:
        hybrid_samples = _robust_monomer_grms(
            stats.hybrid_per_monomer, default=charmm_p90
        )
        hybrid_p90 = _percentile_safe(hybrid_samples, 90.0, charmm_p90)
        hybrid_max = float(np.max(hybrid_samples))

    intervention = max(
        5.0,
        2.5 * charmm_p90,
        2.0 * charmm_max,
        float(np.sqrt(n_mol)) * 0.75,
    )
    if hybrid_p90 is not None:
        intervention = max(
            intervention,
            1.5 * hybrid_p90,
            1.2 * _percentile_safe(hybrid_samples, 75.0, hybrid_p90),
            3.0 * charmm_p90,
        )

    hybrid_total = stats.hybrid_total
    if (
        hybrid_total is not None
        and np.isfinite(hybrid_total)
        and float(hybrid_total) > charmm_bonded_ok_max
        and float(hybrid_total) > 5.0 * max(float(stats.charmm_total), 1.0e-3)
    ):
        # ML geometry stress: cap intervention near the live hybrid total so
        # density-prep / calculator mini run before dynamics (per-monomer tails
        # from registration can otherwise set intervention in the thousands).
        stress_cap = max(25.0, 0.85 * float(hybrid_total))
        intervention = stress_cap
    elif (
        hybrid_total is not None
        and hybrid_max is not None
        and np.isfinite(hybrid_total)
        and float(hybrid_max) > 3.0 * float(hybrid_total)
    ):
        intervention = min(intervention, max(25.0, 0.85 * float(hybrid_total)))

    legacy_from_mol = float(n_mol) * 0.75
    legacy_from_atoms = float(n_at) * 0.2
    max_dyn = max(
        float(base_max_grms),
        intervention * 2.5,
        legacy_from_mol,
        legacy_from_atoms,
    )
    if pbc:
        max_dyn = max(max_dyn, min(250.0, float(n_mol) * 0.85))
    if hybrid_max is not None and hybrid_max > intervention:
        max_dyn = max(max_dyn, min(hybrid_max * 0.35, 250.0))

    notes = (
        f"charmm_p90={charmm_p90:.2f}, charmm_max={charmm_max:.2f}"
        + (
            f", hybrid_p90={hybrid_p90:.2f}, hybrid_max={hybrid_max:.2f}"
            if hybrid_p90 is not None and hybrid_max is not None
            else ""
        )
    )
    return GrmsThresholds(
        intervention_grms=float(intervention),
        max_grms_before_dyn=float(max_dyn),
        charmm_p90=float(charmm_p90),
        hybrid_p90=hybrid_p90,
        notes=notes,
    )


def resolve_grms_thresholds(
    args: argparse.Namespace,
    *,
    atoms_per_list: list[int] | tuple[int, ...] | None,
    n_monomers: int,
    n_atoms: int,
    pbc: bool = False,
    mlpot_ctx: Any | None = None,
) -> GrmsThresholds:
    """Resolve intervention and dynamics GRMS thresholds for the current geometry."""
    base = float(getattr(args, "max_grms_before_dyn", 50.0))
    if atoms_per_list is None:
        n_mol = max(1, int(n_monomers))
        n_at = max(1, int(n_atoms))
        if n_mol > 1 and n_at % n_mol == 0:
            atoms_per_list = [n_at // n_mol] * n_mol
        else:
            atoms_per_list = [n_at]

    stats = measure_monomer_grms_stats(list(atoms_per_list), mlpot_ctx=mlpot_ctx)
    thresholds = resolve_grms_thresholds_from_stats(
        stats,
        n_monomers=n_monomers,
        n_atoms=n_atoms,
        pbc=pbc,
        base_max_grms=base,
    )

    explicit_intervention = getattr(args, "intervention_grms_kcalmol_A", None)
    if explicit_intervention is not None:
        intervention = float(explicit_intervention)
    else:
        intervention = thresholds.intervention_grms

    if getattr(args, "no_scale_max_grms", False) or getattr(args, "allow_high_grms", False):
        max_dyn = base
    else:
        max_dyn = thresholds.max_grms_before_dyn

    if not getattr(args, "quiet", False):
        print(
            f"GRMS thresholds: intervention={intervention:.1f}, "
            f"max_before_dyn={max_dyn:.1f} kcal/mol/Å "
            f"({thresholds.notes})",
            flush=True,
        )
    return GrmsThresholds(
        intervention_grms=float(intervention),
        max_grms_before_dyn=float(max_dyn),
        charmm_p90=thresholds.charmm_p90,
        hybrid_p90=thresholds.hybrid_p90,
        notes=thresholds.notes,
    )


def resolve_intervention_grms_threshold(
    args: argparse.Namespace,
    *,
    atoms_per_list: list[int] | tuple[int, ...] | None,
    n_monomers: int,
    n_atoms: int,
    mlpot_ctx: Any | None = None,
    pbc: bool = False,
) -> float:
    """Low bar for cheap repack/MC before expensive hybrid minimization."""
    return resolve_grms_thresholds(
        args,
        atoms_per_list=atoms_per_list,
        n_monomers=n_monomers,
        n_atoms=n_atoms,
        pbc=pbc,
        mlpot_ctx=mlpot_ctx,
    ).intervention_grms


def resolve_max_grms_before_dyn_intelligent(
    args: argparse.Namespace,
    n_monomers: int,
    n_atoms: int,
    *,
    pbc: bool = False,
    mlpot_ctx: Any | None = None,
    atoms_per_list: list[int] | tuple[int, ...] | None = None,
) -> float:
    """Size-aware dynamics GRMS ceiling using per-monomer MM/hybrid statistics."""
    return resolve_grms_thresholds(
        args,
        atoms_per_list=atoms_per_list,
        n_monomers=n_monomers,
        n_atoms=n_atoms,
        pbc=pbc,
        mlpot_ctx=mlpot_ctx,
    ).max_grms_before_dyn
