"""JAX vs PyCHARMM recovery-MM parity dashboard (bonded + VDW, ELEC off)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.utils.rich_report import emit_dashboard


@dataclass(frozen=True)
class RecoveryMmParityMetrics:
    jax_bonded_kcal: float
    charmm_bonded_kcal: float
    jax_vdw_kcal: float
    charmm_vdw_kcal: float
    jax_total_kcal: float
    charmm_total_kcal: float
    delta_energy_kcal: float
    force_rms_delta: float
    force_max_delta: float
    n_atoms_compared: int
    within_tolerance: bool


def _recovery_nbond_settings(ctx: Any):
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        PBC_CUTNB,
        PBC_CTONNB,
        PBC_CTOFNB,
        VACUUM_CTONNB,
        VACUUM_CTOFNB,
        VACUUM_CUTNB,
        pbc_nbond_cutoffs,
    )

    if bool(getattr(ctx, "use_pbc", False)) and getattr(ctx, "cubic_box_side_A", None):
        cuts = pbc_nbond_cutoffs(float(ctx.cubic_box_side_A))
        return CharmmNbondSettings(
            cutnb=float(cuts.cutnb),
            ctonnb=float(cuts.ctonnb),
            ctofnb=float(cuts.ctofnb),
        )
    return CharmmNbondSettings(
        cutnb=float(VACUUM_CUTNB),
        ctonnb=float(VACUUM_CTONNB),
        ctofnb=float(VACUUM_CTOFNB),
    )


def _cell_from_ctx(ctx: Any) -> np.ndarray:
    if bool(getattr(ctx, "use_pbc", False)) and getattr(ctx, "cubic_box_side_A", None):
        side = float(ctx.cubic_box_side_A)
        return np.diag([side, side, side]).astype(np.float64)
    return np.diag([1000.0, 1000.0, 1000.0]).astype(np.float64)


def _charmm_recovery_reference() -> tuple[dict[str, float], np.ndarray]:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        charmm_bonded_energy_components_kcalmol,
        charmm_bonded_forces_kcalmol_A,
        charmm_nonbonded_energy_components_kcalmol,
        run_charmm_bonded_ener_force,
    )

    run_charmm_bonded_ener_force(silent=True)
    bonded = charmm_bonded_energy_components_kcalmol()
    nb = charmm_nonbonded_energy_components_kcalmol()
    forces = charmm_bonded_forces_kcalmol_A()
    bonded_total = float(bonded.get("total", 0.0))
    vdw = float(nb.get("vdw", 0.0))
    return (
        {
            "bonded": bonded_total,
            "vdw": vdw,
            "total": bonded_total + vdw,
        },
        forces,
    )


def _jax_recovery_reference(
    ctx: Any,
    positions: np.ndarray,
    *,
    topology_psf: Path | str | None,
    ml_atom_indices: Sequence[int],
) -> tuple[dict[str, float], np.ndarray]:
    from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
    from mmml.interfaces.pycharmmInterface.mm_system_energy import (
        load_nonbonded_system_from_charmm,
        nonbonded_energy_and_forces,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        load_bonded_system_for_recovery,
    )

    import jax.numpy as jnp

    system, psf_source = load_bonded_system_for_recovery(
        ctx,
        positions,
        topology_psf=topology_psf,
        ml_atom_indices=ml_atom_indices,
    )
    try:
        bonded_comp, bonded_forces = bonded_energy_and_forces(
            jnp.asarray(positions),
            system.topology,
            system.bonded,
            urey_k=system.urey_k,
            urey_r0=system.urey_r0,
            energy_unit="kcal/mol",
        )
        nbond_data = load_nonbonded_system_from_charmm(psf_source.path, CGENFF_PRM)
        settings = _recovery_nbond_settings(ctx)
        nb_comp, nb_forces = nonbonded_energy_and_forces(
            positions,
            nbond_data,
            _cell_from_ctx(ctx),
            settings,
        )
        forces = np.asarray(bonded_forces + nb_forces, dtype=np.float64)
        bonded_total = float(bonded_comp["total"])
        vdw = float(nb_comp["vdw"])
        return (
            {
                "bonded": bonded_total,
                "vdw": vdw,
                "total": bonded_total + vdw,
            },
            forces,
        )
    finally:
        psf_source.cleanup()


def collect_recovery_mm_parity_metrics(
    ctx: Any,
    positions: np.ndarray,
    *,
    topology_psf: Path | str | None = None,
    ml_atom_indices: Sequence[int] | None = None,
    energy_atol: float = 0.05,
    force_rms_atol: float = 0.05,
) -> RecoveryMmParityMetrics | None:
    """Compare JAX bonded+VDW vs active CHARMM bonded+VDW BLOCK reference."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        _ml_atom_indices,
    )

    if ml_atom_indices is None:
        ml_atom_indices = _ml_atom_indices(ctx)
    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0:
        return None

    charmm_terms, charmm_forces = _charmm_recovery_reference()
    jax_terms, jax_forces = _jax_recovery_reference(
        ctx,
        pos,
        topology_psf=topology_psf,
        ml_atom_indices=ml_atom_indices,
    )
    n = min(jax_forces.shape[0], charmm_forces.shape[0])
    delta_f = np.asarray(jax_forces[:n], dtype=np.float64) - np.asarray(
        charmm_forces[:n], dtype=np.float64
    )
    force_rms = float(np.sqrt(np.mean(np.sum(delta_f * delta_f, axis=-1))))
    force_max = float(np.max(np.linalg.norm(delta_f, axis=-1)))
    delta_e = float(jax_terms["total"] - charmm_terms["total"])
    within = abs(delta_e) <= energy_atol and force_rms <= force_rms_atol
    return RecoveryMmParityMetrics(
        jax_bonded_kcal=float(jax_terms["bonded"]),
        charmm_bonded_kcal=float(charmm_terms["bonded"]),
        jax_vdw_kcal=float(jax_terms["vdw"]),
        charmm_vdw_kcal=float(charmm_terms["vdw"]),
        jax_total_kcal=float(jax_terms["total"]),
        charmm_total_kcal=float(charmm_terms["total"]),
        delta_energy_kcal=delta_e,
        force_rms_delta=force_rms,
        force_max_delta=force_max,
        n_atoms_compared=n,
        within_tolerance=within,
    )


def emit_recovery_mm_parity_dashboard(
    metrics: RecoveryMmParityMetrics,
    *,
    context: str = "bonded+VDW recovery",
    quiet: bool = False,
) -> None:
    """Rich panel: JAX vs CHARMM bonded+VDW parity before recovery minimization."""
    verdict = "PASS (1:1 within tolerance)" if metrics.within_tolerance else "WARN (mismatch)"
    border = "green" if metrics.within_tolerance else "yellow"
    emit_dashboard(
        f"JAX vs CHARMM — {context}",
        [
            (
                "Bonded energy (kcal/mol)",
                {
                    "JAX": f"{metrics.jax_bonded_kcal:.6f}",
                    "CHARMM": f"{metrics.charmm_bonded_kcal:.6f}",
                    "Δ": f"{metrics.jax_bonded_kcal - metrics.charmm_bonded_kcal:+.6f}",
                },
            ),
            (
                "VDW energy (kcal/mol)",
                {
                    "JAX": f"{metrics.jax_vdw_kcal:.6f}",
                    "CHARMM": f"{metrics.charmm_vdw_kcal:.6f}",
                    "Δ": f"{metrics.jax_vdw_kcal - metrics.charmm_vdw_kcal:+.6f}",
                },
            ),
            (
                "Total bonded+VDW (kcal/mol)",
                {
                    "JAX": f"{metrics.jax_total_kcal:.6f}",
                    "CHARMM": f"{metrics.charmm_total_kcal:.6f}",
                    "Δ": f"{metrics.delta_energy_kcal:+.6f}",
                },
            ),
            (
                "Forces (kcal/mol/Å)",
                {
                    "RMS Δ": f"{metrics.force_rms_delta:.6f}",
                    "max |Δ|": f"{metrics.force_max_delta:.6f}",
                    "atoms": metrics.n_atoms_compared,
                },
            ),
            (
                "Verdict",
                {"status": verdict},
            ),
        ],
        border_style=border,
        quiet=quiet,
    )


def maybe_emit_recovery_mm_parity(
    ctx: Any,
    *,
    topology_psf: Path | str | None = None,
    context: str = "bonded+VDW recovery",
    quiet: bool = False,
) -> RecoveryMmParityMetrics | None:
    """Collect and print parity metrics for the current CHARMM coordinates."""
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _mlpot_covers_all_atoms,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    if _mlpot_covers_all_atoms(ctx):
        return None
    positions = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    metrics = collect_recovery_mm_parity_metrics(
        ctx,
        positions,
        topology_psf=topology_psf,
    )
    if metrics is None:
        return None
    emit_recovery_mm_parity_dashboard(metrics, context=context, quiet=quiet)
    return metrics
