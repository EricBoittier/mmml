"""Per-monomer velocity / force / energy bookkeeping and early template restore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.grms_thresholds import (
    measure_monomer_grms_stats,
)

MonomerHealthLevel = Literal["ok", "warn", "bad"]
ComponentName = Literal["velocity", "force", "energy"]

LEVEL_OK: MonomerHealthLevel = "ok"
LEVEL_WARN: MonomerHealthLevel = "warn"
LEVEL_BAD: MonomerHealthLevel = "bad"

_DOT_PLAIN = {LEVEL_OK: "G", LEVEL_WARN: "O", LEVEL_BAD: "R"}
_DOT_RICH = {
    LEVEL_OK: "[green]●[/green]",
    LEVEL_WARN: "[yellow]◐[/yellow]",
    LEVEL_BAD: "[red]○[/red]",
}


@dataclass(frozen=True)
class MonomerHealthConfig:
    """Thresholds for per-monomer early intervention during dynamics."""

    enabled: bool = True
    debug_dot_matrix: bool = False
    template_restore_on_bad: bool = True
    per_monomer_jax_after_restore: bool = True
    max_restore_per_check: int = 4
    velocity_warn_ratio: float = 3.0
    velocity_bad_ratio: float = 6.0
    velocity_warn_abs_akma: float = 5000.0
    velocity_bad_abs_akma: float = 15000.0
    force_warn_ratio: float = 2.5
    force_bad_ratio: float = 5.0
    force_warn_abs_kcalmol_A: float = 30.0
    force_bad_abs_kcalmol_A: float = 80.0
    energy_warn_ratio: float = 2.0
    energy_bad_ratio: float = 4.0
    energy_warn_abs_kcalmol_A: float = 25.0
    energy_bad_abs_kcalmol_A: float = 60.0
    verbose: bool = False


@dataclass(frozen=True)
class MonomerHealthBaseline:
    """Reference per-monomer metrics recorded at dynamics start or first audit."""

    velocity_rms_akma: np.ndarray
    velocity_max_akma: np.ndarray
    hybrid_grms_kcalmol_A: np.ndarray
    charmm_grms_kcalmol_A: np.ndarray
    global_step: int | None = None


@dataclass(frozen=True)
class MonomerHealthEntry:
    """Health snapshot for one monomer / residue."""

    index: int
    label: str
    velocity_rms_akma: float | None
    velocity_max_akma: float | None
    hybrid_grms_kcalmol_A: float | None
    charmm_grms_kcalmol_A: float | None
    velocity_level: MonomerHealthLevel
    force_level: MonomerHealthLevel
    energy_level: MonomerHealthLevel
    reasons: tuple[str, ...] = ()

    @property
    def overall_level(self) -> MonomerHealthLevel:
        levels = (self.velocity_level, self.force_level, self.energy_level)
        if LEVEL_BAD in levels:
            return LEVEL_BAD
        if LEVEL_WARN in levels:
            return LEVEL_WARN
        return LEVEL_OK

    @property
    def needs_template_restore(self) -> bool:
        return self.overall_level == LEVEL_BAD


@dataclass(frozen=True)
class MonomerHealthReport:
  entries: tuple[MonomerHealthEntry, ...]
  flagged_bad: tuple[int, ...]
  flagged_warn: tuple[int, ...]
  baseline_recorded: bool
  restored: bool = False


def monomer_health_config_from_args(args: Any | None) -> MonomerHealthConfig:
    if args is None:
        return MonomerHealthConfig()
    return MonomerHealthConfig(
        enabled=not bool(getattr(args, "no_dynamics_monomer_health", False)),
        debug_dot_matrix=bool(getattr(args, "dynamics_monomer_health_debug", False)),
        template_restore_on_bad=not bool(
            getattr(args, "no_dynamics_monomer_template_restore", False)
        ),
        per_monomer_jax_after_restore=not bool(
            getattr(args, "no_dynamics_monomer_jax_after_restore", False)
        ),
        max_restore_per_check=max(
            1, int(getattr(args, "dynamics_monomer_health_max_restore", 4) or 4)
        ),
        velocity_warn_ratio=float(
            getattr(args, "dynamics_monomer_velocity_warn_ratio", 3.0) or 3.0
        ),
        velocity_bad_ratio=float(
            getattr(args, "dynamics_monomer_velocity_bad_ratio", 6.0) or 6.0
        ),
        velocity_warn_abs_akma=float(
            getattr(args, "dynamics_monomer_velocity_warn_akma", 5000.0) or 5000.0
        ),
        velocity_bad_abs_akma=float(
            getattr(args, "dynamics_monomer_velocity_bad_akma", 15000.0) or 15000.0
        ),
        force_warn_ratio=float(
            getattr(args, "dynamics_monomer_force_warn_ratio", 2.5) or 2.5
        ),
        force_bad_ratio=float(
            getattr(args, "dynamics_monomer_force_bad_ratio", 5.0) or 5.0
        ),
        energy_warn_ratio=float(
            getattr(args, "dynamics_monomer_energy_warn_ratio", 2.0) or 2.0
        ),
        energy_bad_ratio=float(
            getattr(args, "dynamics_monomer_energy_bad_ratio", 4.0) or 4.0
        ),
        verbose=not bool(getattr(args, "quiet", False)),
    )


def resolve_monomer_offsets_for_ctx(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    n_atoms: int,
) -> np.ndarray | None:
    """Return cumulative monomer offsets, preferring PSF-derived atom counts."""
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import monomer_offsets

    atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per is None:
        pyCModel = getattr(mlpot_ctx, "pyCModel", None)
        if pyCModel is not None:
            atoms_per = getattr(pyCModel, "_atoms_per_monomer", None)
    if atoms_per is not None:
        try:
            per = [int(x) for x in atoms_per]
        except TypeError:
            per = []
        if per and int(sum(per)) == int(n_atoms) and len(per) == int(n_monomers):
            return monomer_offsets_from_atoms_per(per)
    if int(n_monomers) > 0 and int(n_atoms) % int(n_monomers) == 0:
        return monomer_offsets(int(n_atoms), int(n_monomers))
    return None


def _per_monomer_velocity_stats(
    velocities_akma: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    n_monomers = int(len(offsets) - 1)
    rms = np.empty(n_monomers, dtype=np.float64)
    vmax = np.empty(n_monomers, dtype=np.float64)
    for mi in range(n_monomers):
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        speeds = np.linalg.norm(v[s:e], axis=1)
        rms[mi] = float(np.sqrt(np.mean(speeds * speeds))) if speeds.size else 0.0
        vmax[mi] = float(np.max(speeds)) if speeds.size else 0.0
    return rms, vmax


def _read_velocities_akma(n_atoms: int) -> np.ndarray | None:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
            charmm_synced_velocities_akma,
        )

        vel = charmm_synced_velocities_akma()
        if vel is not None and int(vel.shape[0]) >= int(n_atoms):
            return np.asarray(vel[:n_atoms], dtype=np.float64)
    except Exception:
        pass
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
            _charmm_velocities_array,
        )

        vel = _charmm_velocities_array()
        if vel is not None and int(vel.shape[0]) >= int(n_atoms):
            return np.asarray(vel[:n_atoms], dtype=np.float64)
    except Exception:
        pass
    return None


def _classify_component(
    value: float | None,
    baseline: float | None,
    *,
    warn_ratio: float,
    bad_ratio: float,
    warn_abs: float,
    bad_abs: float,
    name: str,
) -> tuple[MonomerHealthLevel, tuple[str, ...]]:
    if value is None or not np.isfinite(value):
        return LEVEL_OK, ()
    reasons: list[str] = []
    level = LEVEL_OK
    val = float(value)
    if val >= float(bad_abs):
        level = LEVEL_BAD
        reasons.append(f"{name} abs {val:.1f} ≥ {bad_abs:.1f}")
    elif val >= float(warn_abs):
        if _level_rank(level) < _level_rank(LEVEL_WARN):
            level = LEVEL_WARN
        reasons.append(f"{name} abs {val:.1f} ≥ {warn_abs:.1f}")
    if baseline is not None and np.isfinite(baseline) and float(baseline) > 1.0e-8:
        ratio = val / float(baseline)
        if ratio >= float(bad_ratio):
            level = LEVEL_BAD
            reasons.append(f"{name} ratio {ratio:.1f}× baseline")
        elif ratio >= float(warn_ratio):
            if _level_rank(level) < _level_rank(LEVEL_WARN):
                level = LEVEL_WARN
            reasons.append(f"{name} ratio {ratio:.1f}× baseline")
    return level, tuple(reasons)


def _level_rank(level: MonomerHealthLevel) -> int:
    return {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_BAD: 2}[level]


def collect_monomer_health_metrics(
    mlpot_ctx: Any,
    offsets: np.ndarray,
    *,
    n_monomers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (vel_rms, vel_max, hybrid_grms, charmm_grms) per monomer."""
    n_atoms = int(offsets[-1])
    vel_rms = np.zeros(int(n_monomers), dtype=np.float64)
    vel_max = np.zeros(int(n_monomers), dtype=np.float64)
    vel = _read_velocities_akma(n_atoms)
    if vel is not None:
        vel_rms, vel_max = _per_monomer_velocity_stats(vel, offsets)

    atoms_per = [int(offsets[i + 1] - offsets[i]) for i in range(int(n_monomers))]
    stats = measure_monomer_grms_stats(atoms_per, mlpot_ctx=mlpot_ctx)
    hybrid = (
        np.asarray(stats.hybrid_per_monomer, dtype=np.float64)
        if stats.hybrid_per_monomer is not None
        else np.full(int(n_monomers), np.nan, dtype=np.float64)
    )
    charmm = np.asarray(stats.charmm_per_monomer, dtype=np.float64)
    return vel_rms, vel_max, hybrid, charmm


def record_monomer_health_baseline(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    global_step: int | None = None,
) -> MonomerHealthBaseline | None:
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    n_atoms = int(coor.get_natom())
    offsets = resolve_monomer_offsets_for_ctx(
        mlpot_ctx, n_monomers=int(n_monomers), n_atoms=n_atoms
    )
    if offsets is None:
        return None
    vel_rms, vel_max, hybrid, charmm = collect_monomer_health_metrics(
        mlpot_ctx, offsets, n_monomers=int(n_monomers)
    )
    baseline = MonomerHealthBaseline(
        velocity_rms_akma=vel_rms,
        velocity_max_akma=vel_max,
        hybrid_grms_kcalmol_A=hybrid,
        charmm_grms_kcalmol_A=charmm,
        global_step=global_step,
    )
    setattr(mlpot_ctx, "_monomer_health_baseline", baseline)
    return baseline


def _get_baseline(mlpot_ctx: Any) -> MonomerHealthBaseline | None:
    baseline = getattr(mlpot_ctx, "_monomer_health_baseline", None)
    if isinstance(baseline, MonomerHealthBaseline):
        return baseline
    return None


def audit_monomer_health(
    mlpot_ctx: Any,
    config: MonomerHealthConfig,
    *,
    n_monomers: int,
    global_step: int | None = None,
) -> MonomerHealthReport | None:
    """Classify per-monomer velocity / force / energy stress vs baseline."""
    if not config.enabled or int(n_monomers) <= 1:
        return None

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    n_atoms = int(coor.get_natom())
    offsets = resolve_monomer_offsets_for_ctx(
        mlpot_ctx, n_monomers=int(n_monomers), n_atoms=n_atoms
    )
    if offsets is None:
        return None

    baseline = _get_baseline(mlpot_ctx)
    if baseline is None:
        record_monomer_health_baseline(
            mlpot_ctx, n_monomers=int(n_monomers), global_step=global_step
        )
        return MonomerHealthReport(
            entries=(),
            flagged_bad=(),
            flagged_warn=(),
            baseline_recorded=True,
        )

    vel_rms, vel_max, hybrid, charmm = collect_monomer_health_metrics(
        mlpot_ctx, offsets, n_monomers=int(n_monomers)
    )
    from mmml.interfaces.pycharmmInterface.cluster_geometry import (
        resolve_cluster_residue_labels,
    )

    labels = resolve_cluster_residue_labels(mlpot_ctx, int(n_monomers))
    entries: list[MonomerHealthEntry] = []
    bad: list[int] = []
    warn: list[int] = []

    for mi in range(int(n_monomers)):
        v_level, v_reasons = _classify_component(
            float(vel_max[mi]) if np.isfinite(vel_max[mi]) else None,
            float(baseline.velocity_max_akma[mi])
            if mi < baseline.velocity_max_akma.size
            else None,
            warn_ratio=config.velocity_warn_ratio,
            bad_ratio=config.velocity_bad_ratio,
            warn_abs=config.velocity_warn_abs_akma,
            bad_abs=config.velocity_bad_abs_akma,
            name="|v|",
        )
        f_val = float(hybrid[mi]) if np.isfinite(hybrid[mi]) else float(charmm[mi])
        f_base = (
            float(baseline.hybrid_grms_kcalmol_A[mi])
            if mi < baseline.hybrid_grms_kcalmol_A.size
            and np.isfinite(baseline.hybrid_grms_kcalmol_A[mi])
            else (
                float(baseline.charmm_grms_kcalmol_A[mi])
                if mi < baseline.charmm_grms_kcalmol_A.size
                else None
            )
        )
        f_level, f_reasons = _classify_component(
            f_val if np.isfinite(f_val) else None,
            f_base,
            warn_ratio=config.force_warn_ratio,
            bad_ratio=config.force_bad_ratio,
            warn_abs=config.force_warn_abs_kcalmol_A,
            bad_abs=config.force_bad_abs_kcalmol_A,
            name="GRMS",
        )
        e_val = float(charmm[mi]) if np.isfinite(charmm[mi]) else f_val
        e_base = (
            float(baseline.charmm_grms_kcalmol_A[mi])
            if mi < baseline.charmm_grms_kcalmol_A.size
            else f_base
        )
        e_level, e_reasons = _classify_component(
            e_val if np.isfinite(e_val) else None,
            e_base,
            warn_ratio=config.energy_warn_ratio,
            bad_ratio=config.energy_bad_ratio,
            warn_abs=config.energy_warn_abs_kcalmol_A,
            bad_abs=config.energy_bad_abs_kcalmol_A,
            name="MM",
        )
        reasons = v_reasons + f_reasons + e_reasons
        entry = MonomerHealthEntry(
            index=int(mi),
            label=str(labels[mi]) if mi < len(labels) else f"M{mi:02d}",
            velocity_rms_akma=float(vel_rms[mi]) if np.isfinite(vel_rms[mi]) else None,
            velocity_max_akma=float(vel_max[mi]) if np.isfinite(vel_max[mi]) else None,
            hybrid_grms_kcalmol_A=float(hybrid[mi]) if np.isfinite(hybrid[mi]) else None,
            charmm_grms_kcalmol_A=float(charmm[mi]) if np.isfinite(charmm[mi]) else None,
            velocity_level=v_level,
            force_level=f_level,
            energy_level=e_level,
            reasons=reasons,
        )
        entries.append(entry)
        if entry.overall_level == LEVEL_BAD:
            bad.append(int(mi))
        elif entry.overall_level == LEVEL_WARN:
            warn.append(int(mi))

    return MonomerHealthReport(
        entries=tuple(entries),
        flagged_bad=tuple(bad),
        flagged_warn=tuple(warn),
        baseline_recorded=False,
    )


def emit_monomer_health_dot_matrix(
    report: MonomerHealthReport,
    *,
    context: str,
    quiet: bool = False,
) -> None:
    """Print residue grid: green/yellow/red dots for velocity, force, energy."""
    if not report.entries:
        return
    from mmml.utils.rich_report import emit, rich_enabled

    use_rich = rich_enabled(quiet=quiet)
    header = "Monomer health  [v]=velocity [f]=force [e]=MM/energy"
    lines = [header, " idx   v f e  residue"]
    for entry in report.entries:
        if use_rich:
            v_dot = _DOT_RICH[entry.velocity_level]
            f_dot = _DOT_RICH[entry.force_level]
            e_dot = _DOT_RICH[entry.energy_level]
        else:
            v_dot = _DOT_PLAIN[entry.velocity_level]
            f_dot = _DOT_PLAIN[entry.force_level]
            e_dot = _DOT_PLAIN[entry.energy_level]
        dots = f"{v_dot} {f_dot} {e_dot}"
        lines.append(
            f" {entry.index:3d}  {dots}  {entry.label}"
            + (f"  ({'; '.join(entry.reasons)})" if entry.reasons else "")
        )
    body = "\n".join(lines)
    if use_rich:
        emit(f"[bold]{context}[/bold]\n{body}", quiet=quiet)
    else:
        emit(f"{context}\n{body}", quiet=quiet)


def restore_flagged_monomers_from_template(
    mlpot_ctx: Any,
    flagged: tuple[int, ...] | list[int],
    *,
    context: str,
    restart_path: Any | None = None,
    verbose: bool = False,
) -> bool:
    """Rigid-body template restore for unhealthy monomers only."""
    if not flagged:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini import (
        remember_monomer_template_restart_path,
        resolve_monomer_template_reference_positions,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )
    from mmml.utils.geometry_checks import rebuild_monomers_from_reference

    remember_monomer_template_restart_path(mlpot_ctx, restart_path)
    n_atoms = int(np.asarray(get_charmm_positions_array()).shape[0])
    atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per is None:
        pyCModel = getattr(mlpot_ctx, "pyCModel", None)
        atoms_per = getattr(pyCModel, "_atoms_per_monomer", None) if pyCModel else None
    if not atoms_per:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )

    offsets = monomer_offsets_from_atoms_per([int(x) for x in atoms_per])
    ref_info = resolve_monomer_template_reference_positions(
        mlpot_ctx, restart_path=restart_path, n_atoms=n_atoms
    )
    if ref_info is None:
        if verbose:
            print(f"WARN: {context}: no template reference for monomer restore", flush=True)
        return False
    ref, source = ref_info
    pos = np.asarray(get_charmm_positions_array(), dtype=np.float64)
    selected = [int(i) for i in flagged]
    new_pos = rebuild_monomers_from_reference(pos, ref, offsets, selected)
    sync_charmm_positions(new_pos)
    if verbose:
        print(
            f"{context}: template-restored monomer(s) "
            f"{selected} from {source.name}",
            flush=True,
        )
    return True


def _run_per_monomer_jax_on_indices(
    mlpot_ctx: Any,
    overlap_config: Any,
    monomer_indices: tuple[int, ...],
    *,
    context: str,
) -> None:
    if not monomer_indices:
        return
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_jax_recovery import (
        minimize_bonded_jax_per_monomer_recovery,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.bonded_mm_recovery import (
        _bonded_cfg_from_overlap_config,
    )

    bonded_cfg = _bonded_cfg_from_overlap_config(overlap_config)
    topo = getattr(overlap_config, "topology_psf", None) or getattr(
        mlpot_ctx, "topology_psf_path", None
    )
    minimize_bonded_jax_per_monomer_recovery(
        mlpot_ctx,
        bonded_cfg,
        n_monomers=int(getattr(overlap_config, "n_monomers", 1) or 1),
        topology_psf=topo,
        context=f"{context} (per-monomer JAX)",
        monomer_indices=monomer_indices,
    )


def maybe_intervene_monomer_health(
    mlpot_ctx: Any,
    overlap_config: Any,
    *,
    context: str,
    global_step: int | None = None,
    restart_path: Any | None = None,
) -> bool:
    """Audit per-monomer health; template-restore + JAX mini on bad monomers.

    Returns True when coordinates were rewritten (caller should refresh restart).
    """
    health_cfg = getattr(overlap_config, "monomer_health", None)
    if health_cfg is None:
        args = getattr(mlpot_ctx, "workflow_args", None)
        health_cfg = monomer_health_config_from_args(args)
    if not health_cfg.enabled:
        return False

    n_monomers = int(getattr(overlap_config, "n_monomers", 1) or 1)
    report = audit_monomer_health(
        mlpot_ctx,
        health_cfg,
        n_monomers=n_monomers,
        global_step=global_step,
    )
    if report is None or report.baseline_recorded:
        return False

    if health_cfg.debug_dot_matrix or report.flagged_bad or report.flagged_warn:
        emit_monomer_health_dot_matrix(
            report,
            context=f"{context} monomer health (step {global_step})",
            quiet=not health_cfg.verbose and not health_cfg.debug_dot_matrix,
        )

    if not report.flagged_bad:
        return False

    to_restore = report.flagged_bad[: int(health_cfg.max_restore_per_check)]
    if not health_cfg.template_restore_on_bad:
        if health_cfg.verbose:
            print(
                f"{context}: monomer health bad {to_restore} "
                "(template restore disabled)",
                flush=True,
            )
        return False

    restored = restore_flagged_monomers_from_template(
        mlpot_ctx,
        to_restore,
        context=context,
        restart_path=restart_path,
        verbose=health_cfg.verbose or health_cfg.debug_dot_matrix,
    )
    if not restored:
        return False

    if health_cfg.per_monomer_jax_after_restore:
        _run_per_monomer_jax_on_indices(
            mlpot_ctx,
            overlap_config,
            tuple(to_restore),
            context=context,
        )

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
    )

    invalidate_mlpot_calculator_caches(mlpot_ctx)
    mlpot_ctx.reregister_mlpot(verbose=False)
    return True
