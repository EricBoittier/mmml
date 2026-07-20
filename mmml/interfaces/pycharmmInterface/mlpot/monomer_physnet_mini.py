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
    """Relax 1–2 stressed monomers/dimers with an isolated PhysNet calculator."""

    max_select: int = 2
    min_abs_grms: float = 25.0
    min_ratio_to_median: float = 2.5
    max_steps: int = 60
    fmax_ev_a: float = 0.05
    bfgs_maxstep: float = 0.05
    optimize_dimers: bool = True
    verbose: bool = True
    quiet_bfgs: bool = False
    max_grms_regression_ratio: float = 1.02


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
        optimize_dimers=not bool(
            getattr(args, "no_monomer_physnet_mini_dimers", False)
        ),
        verbose=bool(verbose),
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


def build_monomer_template_recovery_candidates(
    mlpot_ctx: Any,
    *,
    restart_path: Path | str | None = None,
) -> list[Path]:
    """Disk restart/CRD ladder for flagged-monomer template restore."""
    from types import SimpleNamespace

    from mmml.interfaces.pycharmmInterface.mlpot.geometry_checkpoint import (
        build_extent_recovery_candidates,
        resolve_geometry_checkpoint_ladder,
    )

    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        p = Path(path).expanduser()
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        candidates.append(p)

    add(restart_path)
    add(getattr(mlpot_ctx, "_monomer_template_restart_path", None))

    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is not None:
        for attr in (
            "restart_path",
            "pretreat_restart",
            "from_crd",
            "mini_crd",
            "geometry_baseline_res",
        ):
            add(getattr(args, attr, None))

        paths = getattr(args, "_artifact_paths", None) or getattr(args, "paths", None)
        if isinstance(paths, dict):
            tag = str(getattr(args, "tag", "") or "")
            n_heat = max(1, int(getattr(args, "n_heat_segments", 1) or 1))
            ladder_paths = resolve_geometry_checkpoint_ladder(
                paths,
                tag,
                n_heat_segments=n_heat,
            )
            baseline = paths.get("geometry_baseline_res")
            overlap_ns = SimpleNamespace(
                geometry_baseline_restart=baseline,
                geometry_fallback_restarts=tuple(ladder_paths),
                prior_segment_restart=(
                    getattr(args, "restart_path", None)
                    or getattr(args, "pretreat_restart", None)
                ),
            )
            for cand in build_extent_recovery_candidates(overlap_ns):
                add(cand)

    return candidates


def resolve_monomer_template_reference_positions(
    mlpot_ctx: Any,
    *,
    restart_path: Path | str | None = None,
    n_atoms: int | None = None,
    current_positions: np.ndarray | None = None,
    monomer_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, Path] | None:
    """Return a plausible monomer template, preferring disk then memory.

    Dynamics restart files can be syntactically readable while containing
    collapsed coordinate-history values.  Reject references whose centered
    monomer radii differ grossly from the intact current geometry and continue
    down the in-memory mini/baseline ladder.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.extent_repack_recovery import (
        _memory_extent_reference_sources,
        resolve_extent_reference_positions,
    )

    candidates = build_monomer_template_recovery_candidates(
        mlpot_ctx,
        restart_path=restart_path,
    )
    refs: list[tuple[np.ndarray, Path]] = []
    try:
        ref, source = resolve_extent_reference_positions(candidates, None)
        refs.append((np.asarray(ref, dtype=np.float64), source))
    except RuntimeError:
        pass
    memory_refs = _memory_extent_reference_sources(mlpot_ctx)
    refs.extend(memory_refs)

    # The live geometry may already be the damaged frame that triggered
    # recovery.  Never use it to disqualify the trusted pre-dynamics mini or
    # baseline snapshots.  Instead, validate disk/fallback templates against
    # the first intact in-memory reference when one exists.
    plausibility_reference = None
    for raw_ref, _source in memory_refs:
        arr = np.asarray(raw_ref, dtype=np.float64)
        if arr.size and np.all(np.isfinite(arr)):
            plausibility_reference = arr
            break

    def plausible(arr: np.ndarray) -> bool:
        comparison = (
            plausibility_reference
            if plausibility_reference is not None
            else current_positions
        )
        if comparison is None or monomer_offsets is None:
            return True
        cur = np.asarray(comparison, dtype=np.float64)
        offsets = np.asarray(monomer_offsets, dtype=int)
        if arr.shape != cur.shape or offsets.size < 2:
            return False
        ratios: list[float] = []
        for mi in range(offsets.size - 1):
            s, e = int(offsets[mi]), int(offsets[mi + 1])
            if e - s < 2:
                continue
            cur_centered = cur[s:e] - cur[s:e].mean(axis=0)
            ref_centered = arr[s:e] - arr[s:e].mean(axis=0)
            cur_rms = float(np.sqrt(np.mean(np.sum(cur_centered**2, axis=1))))
            ref_rms = float(np.sqrt(np.mean(np.sum(ref_centered**2, axis=1))))
            if cur_rms > 1.0e-6:
                ratios.append(ref_rms / cur_rms)
        return bool(ratios) and min(ratios) >= 0.5 and max(ratios) <= 2.0

    for ref_arr, source in refs:
        ref_arr = np.asarray(ref_arr, dtype=np.float64)
        if n_atoms is not None and int(ref_arr.shape[0]) != int(n_atoms):
            continue
        if ref_arr.size == 0 or not np.all(np.isfinite(ref_arr)):
            continue
        if plausible(ref_arr):
            return ref_arr, source
        print(
            f"Monomer template: rejecting internally implausible reference {source}; "
            "falling back to intact in-memory/template geometry",
            flush=True,
        )

    try:
        from mmml.interfaces.pycharmmInterface.cluster_geometry import (
            packmol_template_reference_from_ctx,
            same_residue_cluster_reference_from_ctx,
        )

        ref_arr = packmol_template_reference_from_ctx(mlpot_ctx, n_atoms=n_atoms)
        if ref_arr is not None:
            source = Path("<packmol-monomer-template>")
        else:
            ref_arr = same_residue_cluster_reference_from_ctx(mlpot_ctx, n_atoms=n_atoms)
            if ref_arr is None:
                return None
            source = Path("<same-residue-cluster>")
    except (RuntimeError, ValueError):
        return None
    if n_atoms is not None and int(ref_arr.shape[0]) != int(n_atoms):
        return None
    if ref_arr.size == 0 or not np.all(np.isfinite(ref_arr)):
        return None
    if not plausible(ref_arr):
        return None
    return ref_arr, source


def remember_monomer_template_restart_path(
    mlpot_ctx: Any,
    restart_path: Path | str | None,
) -> None:
    if restart_path is None:
        return
    setattr(
        mlpot_ctx,
        "_monomer_template_restart_path",
        Path(restart_path).expanduser().resolve(),
    )


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


def _selected_atom_indices(
    offsets: np.ndarray,
    monomer_indices: tuple[int, ...],
) -> np.ndarray:
    chunks = [
        np.arange(int(offsets[int(mi)]), int(offsets[int(mi) + 1]), dtype=int)
        for mi in monomer_indices
    ]
    if not chunks:
        return np.asarray([], dtype=int)
    return np.concatenate(chunks)


def _systematic_homogeneous_targets(
    flagged: tuple[int, ...],
    atoms_per_list: list[int],
    atomic_numbers: np.ndarray,
) -> tuple[int, ...]:
    """Return every monomer when the complete homogeneous box is stressed."""
    n_monomers = len(atoms_per_list)
    unique = tuple(dict.fromkeys(int(i) for i in flagged))
    if set(unique) != set(range(n_monomers)) or n_monomers < 3:
        return ()
    if len(set(atoms_per_list)) != 1:
        return ()
    atoms_per = int(atoms_per_list[0])
    z = np.asarray(atomic_numbers, dtype=int)
    reference = z[:atoms_per]
    if any(
        not np.array_equal(reference, z[mi * atoms_per : (mi + 1) * atoms_per])
        for mi in range(1, n_monomers)
    ):
        return ()
    return tuple(range(n_monomers))


def transfer_internal_geometry_preserving_pose(
    source_initial: np.ndarray,
    source_optimized: np.ndarray,
    target_initial: np.ndarray,
) -> np.ndarray:
    """Transfer optimized internal geometry while preserving target COM/orientation."""
    src0 = np.asarray(source_initial, dtype=np.float64)
    src1 = np.asarray(source_optimized, dtype=np.float64)
    target = np.asarray(target_initial, dtype=np.float64)
    if src0.shape != src1.shape or src0.shape != target.shape:
        raise ValueError("source and target monomer shapes must match")
    src0_centered = src0 - src0.mean(axis=0)
    src1_centered = src1 - src1.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _singular_values, vt = np.linalg.svd(src0_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return src1_centered @ rotation + target.mean(axis=0)


def _cap_flagged_monomers(
    flagged: tuple[int, ...],
    *,
    max_select: int,
    mlpot_ctx: Any,
    atoms_per_list: list[int],
    positions: np.ndarray,
    context_prefix: str,
    verbose: bool,
) -> tuple[int, ...]:
    """Keep at most ``max_select`` monomers, preferring highest hybrid fmax."""
    if max_select <= 0 or not flagged:
        return ()
    unique = tuple(dict.fromkeys(int(i) for i in flagged))
    if len(unique) <= max_select:
        return unique

    scores = np.full(len(unique), -np.inf, dtype=np.float64)
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
            per_monomer_fmax_from_forces,
        )
        from mmml.utils.monomer_force_diag import mlpot_hybrid_forces_kcalmol_A

        forces = mlpot_hybrid_forces_kcalmol_A(mlpot_ctx, positions=positions)
        if forces is not None:
            per_mono = per_monomer_fmax_from_forces(forces, atoms_per_list)
            for j, mi in enumerate(unique):
                if 0 <= int(mi) < int(per_mono.size):
                    scores[j] = float(per_mono[int(mi)])
    except Exception:  # noqa: BLE001 — fall back to input order
        pass

    order = sorted(
        range(len(unique)),
        key=lambda j: (-scores[j], j),
    )
    kept = tuple(unique[j] for j in order[:max_select])
    if verbose:
        print(
            f"{context_prefix}: capping isolated PhysNet mini "
            f"{len(unique)} → {len(kept)} monomer(s) (max_select={max_select}); "
            f"kept [{', '.join(str(i) for i in kept)}]",
            flush=True,
        )
    return kept


def run_selective_monomer_physnet_mini(
    mlpot_ctx: Any,
    *,
    config: SelectiveMonomerPhysnetMiniConfig | None = None,
    context_prefix: str = "Selective monomer PhysNet",
    flagged: tuple[int, ...] | list[int] | None = None,
    restart_path: Path | str | None = None,
) -> SelectiveMonomerPhysnetMiniResult:
    """FIRE-minimize flagged monomers/dimers; rest of the box stays fixed."""
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
    rollback_positions = pos.copy()
    initial_grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
    z_arr = np.asarray(z_full, dtype=int)

    selected: tuple[int, ...]
    systematic_targets: tuple[int, ...] = ()
    if flagged is not None:
        # Explicit lists from pre-dynamics repair can include every monomer when
        # intermolecular force spikes dominate (common in dense liquids). Cap to
        # max_select — vacuum PhysNet on the whole box destroys packing.
        raw_flagged = tuple(int(i) for i in flagged)
        systematic_targets = _systematic_homogeneous_targets(
            raw_flagged,
            atoms_per_list,
            z_arr,
        )
        selected = _cap_flagged_monomers(
            raw_flagged,
            max_select=int(config.max_select),
            mlpot_ctx=mlpot_ctx,
            atoms_per_list=atoms_per_list,
            positions=pos,
            context_prefix=context_prefix,
            verbose=bool(config.verbose),
        )
        if systematic_targets:
            # Optimize one worst representative, then transfer only its
            # internal geometry to the homogeneous box below.
            selected = selected[:1]
            if config.verbose:
                print(
                    f"{context_prefix}: systematic homogeneous stress on "
                    f"{len(systematic_targets)} monomers; optimizing one shared "
                    "internal template while preserving every COM/orientation",
                    flush=True,
                )
        if not selected:
            grms = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
            if grms is None or not np.isfinite(grms):
                grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
            return SelectiveMonomerPhysnetMiniResult(grms=float(grms), ran=False)
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
            from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
                monomer_offsets_from_atoms_per,
            )
            from mmml.utils.monomer_force_diag import (
                format_worst_atom_force_peaks,
                mlpot_hybrid_forces_kcalmol_A,
                worst_atom_force_peaks,
            )

            atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)
            if atoms_per is None:
                pyCModel = getattr(mlpot_ctx, "pyCModel", None)
                atoms_per = (
                    getattr(pyCModel, "_atoms_per_monomer", None) if pyCModel else None
                )
            forces = mlpot_hybrid_forces_kcalmol_A(mlpot_ctx, positions=pos)
            if forces is not None and atoms_per:
                offsets = monomer_offsets_from_atoms_per([int(x) for x in atoms_per])
                peaks = worst_atom_force_peaks(
                    forces,
                    offsets,
                    top_n=3,
                    monomer_filter=selected,
                )
                print(
                    f"{context_prefix}: worst atom |F| before template restore: "
                    f"{format_worst_atom_force_peaks(peaks)}",
                    flush=True,
                )

    remember_monomer_template_restart_path(mlpot_ctx, restart_path)
    ref_info = resolve_monomer_template_reference_positions(
        mlpot_ctx,
        restart_path=restart_path,
        n_atoms=int(pos.shape[0]),
        current_positions=pos,
        monomer_offsets=offsets,
    )
    if ref_info is not None:
        ref, source = ref_info
        pos = rebuild_monomers_from_reference(pos, ref, offsets, list(selected))
        if config.verbose:
            if source.name == "<same-residue-cluster>":
                print(
                    f"{context_prefix}: no external template residue; "
                    f"copying and adjusting positions from same residue type "
                    f"for monomer(s) {list(selected)}",
                    flush=True,
                )
            elif source.name == "<packmol-monomer-template>":
                print(
                    f"{context_prefix}: restored monomer(s) {list(selected)} from "
                    f"CHARMM-minimized Packmol monomer template at current COM",
                    flush=True,
                )
            else:
                print(
                    f"{context_prefix}: restored monomer(s) {list(selected)} from "
                    f"{source} at current COM",
                    flush=True,
                )
    elif config.verbose:
        print(
            f"{context_prefix}: no restart/template reference; "
            f"PhysNet BFGS from current coordinates",
            flush=True,
        )

    checkpoint = resolve_mlpot_checkpoint_path(mlpot_ctx)
    import ase
    import ase.optimize as ase_opt

    logfile: str | io.StringIO = (
        io.StringIO() if config.quiet_bfgs else "-"
    )
    groups: tuple[tuple[int, ...], ...]
    if bool(config.optimize_dimers) and len(selected) == 2:
        groups = (tuple(selected),)
    else:
        groups = tuple((int(mi),) for mi in selected)

    for group in groups:
        atom_idx = _selected_atom_indices(offsets, tuple(int(mi) for mi in group))
        if atom_idx.size == 0:
            continue
        z_sel = z_arr[atom_idx]
        sel_atoms = ase.Atoms(
            numbers=np.asarray(z_sel, dtype=int),
            positions=np.asarray(pos[atom_idx], dtype=np.float64),
        )
        sel_atoms.calc = _monomer_ase_calculator(
            mlpot_ctx,
            checkpoint=checkpoint,
            atomic_numbers=z_sel,
        )
        opt = ase_opt.FIRE(
            sel_atoms,
            maxstep=float(config.bfgs_maxstep),
            logfile=logfile,
        )
        opt.run(fmax=float(config.fmax_ev_a), steps=max(1, int(config.max_steps)))
        pos[atom_idx] = np.asarray(sel_atoms.get_positions(), dtype=np.float64)

    if systematic_targets and selected:
        representative = int(selected[0])
        rep_start, rep_end = int(offsets[representative]), int(offsets[representative + 1])
        source_initial = rollback_positions[rep_start:rep_end]
        source_optimized = pos[rep_start:rep_end]
        for mi in systematic_targets:
            start, end = int(offsets[mi]), int(offsets[mi + 1])
            pos[start:end] = transfer_internal_geometry_preserving_pose(
                source_initial,
                source_optimized,
                rollback_positions[start:end],
            )

    sync_charmm_positions(pos)
    invalidate_mlpot_calculator_caches(mlpot_ctx)
    grms = float(refresh_mlpot_energy_and_grms(mlpot_ctx, context=""))
    rollback_limit = (
        float(initial_grms) * float(config.max_grms_regression_ratio)
        if initial_grms is not None and np.isfinite(initial_grms)
        else 50.0
    )
    if not np.isfinite(grms) or grms > rollback_limit:
        sync_charmm_positions(rollback_positions)
        invalidate_mlpot_calculator_caches(mlpot_ctx)
        restored_grms = refresh_mlpot_energy_and_grms(mlpot_ctx, context="")
        print(
            f"{context_prefix}: rejecting monomer PhysNet result "
            f"(GRMS={grms:.4f} > rollback limit {rollback_limit:.4f}); "
            f"restored pre-recovery geometry (GRMS={restored_grms:.4f})",
            flush=True,
        )
        return SelectiveMonomerPhysnetMiniResult(
            grms=float(restored_grms),
            ran=False,
            flagged=systematic_targets or selected,
        )
    if config.verbose:
        print(
            f"{context_prefix}: done — hybrid GRMS={grms:.4f} kcal/mol/Å "
            f"({len(systematic_targets) if systematic_targets else len(selected)} "
            f"monomer(s), {len(groups)} PhysNet group(s))",
            flush=True,
        )
    return SelectiveMonomerPhysnetMiniResult(
        grms=float(grms),
        ran=True,
        flagged=systematic_targets or selected,
    )
