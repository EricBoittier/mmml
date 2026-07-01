"""Selective monomer PhysNet BFGS on flagged high-force monomers (box frozen)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    mlpot_hybrid_grms_from_calculator,
    refresh_mlpot_energy_and_grms,
    resolve_checkpoint,
)


@dataclass(frozen=True)
class SelectiveMonomerPhysnetMiniResult:
    """Outcome of monomer-only PhysNet relaxation."""

    grms: float
    ran: bool
    flagged: tuple[int, ...] = ()


@dataclass
class SelectiveMonomerPhysnetMiniConfig:
    """Relax 1–2 stressed monomers with an isolated PhysNet calculator."""

    max_select: int = 2
    min_abs_grms: float = 25.0
    min_ratio_to_median: float = 2.5
    max_steps: int = 60
    fmax_ev_a: float = 0.05
    bfgs_maxstep: float = 0.05
    verbose: bool = True
    restore_template: bool = True
    quiet_bfgs: bool = False


def monomer_physnet_mini_enabled(args: Any | None) -> bool:
    if args is None:
        return True
    return bool(getattr(args, "monomer_physnet_mini", True))


def selective_monomer_physnet_mini_config_from_args(
    args: Any | None,
    *,
    verbose: bool = True,
    quiet_bfgs: bool = False,
) -> SelectiveMonomerPhysnetMiniConfig:
    return SelectiveMonomerPhysnetMiniConfig(
        max_select=int(getattr(args, "monomer_physnet_mini_max_select", 2) or 2),
        min_abs_grms=float(getattr(args, "monomer_physnet_mini_min_grms", 25.0) or 25.0),
        min_ratio_to_median=float(
            getattr(args, "monomer_physnet_mini_min_ratio", 2.5) or 2.5
        ),
        max_steps=int(getattr(args, "monomer_physnet_mini_steps", 60) or 60),
        fmax_ev_a=float(
            getattr(args, "monomer_physnet_mini_fmax", None)
            or getattr(args, "pre_min_fmax", 0.05)
            or 0.05
        ),
        bfgs_maxstep=float(
            getattr(args, "monomer_physnet_mini_maxstep", None)
            or getattr(args, "bfgs_maxstep", 0.05)
            or 0.05
        ),
        verbose=bool(verbose),
        restore_template=bool(getattr(args, "monomer_physnet_mini_restore_template", True)),
        quiet_bfgs=bool(quiet_bfgs or getattr(args, "quiet_bfgs", False)),
    )


def resolve_mlpot_checkpoint_path(mlpot_ctx: Any) -> Path:
    args = getattr(mlpot_ctx, "workflow_args", None)
    explicit = getattr(args, "checkpoint", None) if args is not None else None
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    return resolve_checkpoint(None)


def _monomer_offsets(atoms_per_list: list[int]) -> np.ndarray:
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )

    return monomer_offsets_from_atoms_per(atoms_per_list)


def _reference_positions_for_template(mlpot_ctx: Any) -> np.ndarray | None:
    for attr in ("geometry_mini_positions", "geometry_baseline_positions"):
        raw = getattr(mlpot_ctx, attr, None)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.size and np.all(np.isfinite(arr)):
            return arr
    return None


def _monomer_ase_calculator(
    mlpot_ctx: Any,
    *,
    checkpoint: Path,
    atomic_numbers: np.ndarray,
):
    cache = getattr(mlpot_ctx, "_monomer_physnet_calc_cache", None)
    if cache is None:
        cache = {}
        setattr(mlpot_ctx, "_monomer_physnet_calc_cache", cache)
    key = (str(checkpoint), tuple(int(z) for z in np.asarray(atomic_numbers, dtype=int)))
    calc = cache.get(key)
    if calc is not None:
        return calc

    import ase
    from mmml.cli.base import load_physnet_params_and_ef_model
    from mmml.models.physnetjax.physnetjax.calc.helper_mlp import get_ase_calc

    z = np.asarray(atomic_numbers, dtype=int)
    n = int(z.size)
    params, model = load_physnet_params_and_ef_model(checkpoint, natoms=n)
    model.natoms = n
    template = ase.Atoms(numbers=z, positions=np.zeros((n, 3), dtype=float))
    calc = get_ase_calc(params, model, template)
    cache[key] = calc
    return calc


def run_selective_monomer_physnet_mini(
    mlpot_ctx: Any,
    *,
    config: SelectiveMonomerPhysnetMiniConfig | None = None,
    context_prefix: str = "Selective monomer PhysNet",
    flagged: tuple[int, ...] | list[int] | None = None,
) -> SelectiveMonomerPhysnetMiniResult:
    """BFGS flagged monomers with an isolated PhysNet calc; rest of the box stays fixed."""
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )
    from mmml.utils.geometry_checks import rebuild_monomers_from_reference
    from mmml.utils.monomer_force_diag import resolve_selective_repack_monomers

    args = getattr(mlpot_ctx, "workflow_args", None)
    if config is None:
        config = selective_monomer_physnet_mini_config_from_args(
            args,
            verbose=True,
            quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)) if args else False,
        )

    z_full = getattr(mlpot_ctx, "ml_Z", None)
    if z_full is None:
        grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
        return SelectiveMonomerPhysnetMiniResult(
            grms=float(grms) if grms is not None else float("nan"),
            ran=False,
        )

    atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per is None and args is not None:
        atoms_per = getattr(args, "_cluster_atoms_per_list", None)
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    if atoms_per is None and pyCModel is not None:
        atoms_per = getattr(pyCModel, "_atoms_per_monomer", None)
    if atoms_per is None:
        grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
        return SelectiveMonomerPhysnetMiniResult(
            grms=float(grms) if grms is not None else float("nan"),
            ran=False,
        )

    atoms_per_list = [int(x) for x in atoms_per]
    offsets = _monomer_offsets(atoms_per_list)
    pos = np.asarray(get_charmm_positions_array(), dtype=np.float64).copy()
    z_arr = np.asarray(z_full, dtype=int)

    selected: tuple[int, ...]
    if flagged is not None:
        selected = tuple(int(i) for i in flagged)
    else:
        diag = resolve_selective_repack_monomers(
            mlpot_ctx,
            offsets,
            max_select=int(config.max_select),
            min_abs_grms=float(config.min_abs_grms),
            min_ratio_to_median=float(config.min_ratio_to_median),
            positions=pos,
        )
        if diag is None or not diag.flagged:
            grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
            if grms is None or not np.isfinite(grms):
                grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
            return SelectiveMonomerPhysnetMiniResult(grms=float(grms), ran=False)
        selected = tuple(int(i) for i in diag.flagged)
        if config.verbose:
            grms_txt = ", ".join(
                f"{i}:{diag.grms_per_monomer[i]:.1f}" for i in selected
            )
            print(
                f"{context_prefix}: monomer PhysNet BFGS on [{', '.join(str(i) for i in selected)}] "
                f"(per-mono GRMS {grms_txt} kcal/mol/Å; cluster {diag.cluster_grms:.1f})",
                flush=True,
            )

    if config.restore_template:
        ref = _reference_positions_for_template(mlpot_ctx)
        if ref is not None and ref.shape == pos.shape:
            pos = rebuild_monomers_from_reference(pos, ref, offsets, list(selected))
            if config.verbose:
                print(
                    f"{context_prefix}: restored internal template for monomer(s) "
                    f"{list(selected)} at current COM",
                    flush=True,
                )

    checkpoint = resolve_mlpot_checkpoint_path(mlpot_ctx)
    import ase
    import ase.optimize as ase_opt

    logfile: str | io.StringIO = (
        io.StringIO() if config.quiet_bfgs else "-"
    )
    for mi in selected:
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        z_mono = z_arr[s:e]
        mono_atoms = ase.Atoms(
            numbers=np.asarray(z_mono, dtype=int),
            positions=np.asarray(pos[s:e], dtype=np.float64),
        )
        mono_atoms.calc = _monomer_ase_calculator(
            mlpot_ctx,
            checkpoint=checkpoint,
            atomic_numbers=z_mono,
        )
        opt = ase_opt.BFGS(
            mono_atoms,
            maxstep=float(config.bfgs_maxstep),
            logfile=logfile,
        )
        opt.run(fmax=float(config.fmax_ev_a), steps=max(1, int(config.max_steps)))
        pos[s:e] = np.asarray(mono_atoms.get_positions(), dtype=np.float64)

    sync_charmm_positions(pos)
    invalidate_mlpot_calculator_caches(mlpot_ctx)
    grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
    if config.verbose:
        print(
            f"{context_prefix}: done — hybrid GRMS={grms:.4f} kcal/mol/Å "
            f"({len(selected)} monomer(s))",
            flush=True,
        )
    return SelectiveMonomerPhysnetMiniResult(
        grms=float(grms),
        ran=True,
        flagged=selected,
    )
