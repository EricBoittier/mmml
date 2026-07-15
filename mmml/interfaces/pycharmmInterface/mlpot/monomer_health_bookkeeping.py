"""Per-monomer velocity / force / energy bookkeeping and early template restore."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    """Thresholds for per-monomer early intervention during dynamics.

    Template + per-monomer FIRE are reserved for geometry failures (extent,
    bond stretch, COM drift / intra collapse).  Velocity/GRMS stress only
    redraws velocities.  Ratios never escalate a level unless the absolute
    floor is also met, and baselines are floored so a near-zero post-mini
    reference cannot explode.
    """

    enabled: bool = True
    debug_dot_matrix: bool = False
    template_restore_on_bad: bool = True
    per_monomer_jax_after_restore: bool = True
    velocity_restore_on_template: bool = True
    max_restore_per_check: int = 4
    velocity_warn_ratio: float = 3.0
    velocity_bad_ratio: float = 6.0
    velocity_warn_abs_akma: float = 5000.0
    velocity_bad_abs_akma: float = 15000.0
    velocity_warn_recover_fraction: float = 0.80
    force_warn_ratio: float = 2.5
    force_bad_ratio: float = 5.0
    force_warn_abs_kcalmol_A: float = 30.0
    force_bad_abs_kcalmol_A: float = 80.0
    # Kept for CLI compat; MM/energy GRMS double-count is disabled.
    energy_warn_ratio: float = 2.0
    energy_bad_ratio: float = 4.0
    energy_warn_abs_kcalmol_A: float = 25.0
    energy_bad_abs_kcalmol_A: float = 60.0
    # Floor baseline for ratio math as a fraction of the warn absolute cut.
    baseline_floor_fraction_of_warn: float = 0.25
    # Ratio may only annotate / escalate when abs is already at least warn.
    ratio_requires_abs_warn: bool = True
    # Template+FIRE only when geometry fails (not velocity/GRMS alone).
    template_restore_requires_geometry: bool = True
    # Flag PSF 1–2 bonds stretched beyond factor × baseline (or abs floor).
    bond_stretch_factor: float = 1.75
    bond_stretch_abs_A: float = 2.50
    # Incremental COM unwrap drift vs health baseline (catches rigid fly-aways
    # under IMAGE centering that intramolecular extent cannot see).
    com_flyoff_enabled: bool = True
    com_flyoff_A: float = 0.0  # <=0 → max(15 Å, 0.35 * cubic box side)
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
    energy_level: MonomerHealthLevel  # legacy alias; geometry uses geometry_level
    reasons: tuple[str, ...] = ()
    geometry_level: MonomerHealthLevel = LEVEL_OK

    @property
    def overall_level(self) -> MonomerHealthLevel:
        levels = (
            self.velocity_level,
            self.force_level,
            self.geometry_level,
        )
        if LEVEL_BAD in levels:
            return LEVEL_BAD
        if LEVEL_WARN in levels:
            return LEVEL_WARN
        return LEVEL_OK

    @property
    def needs_template_restore(self) -> bool:
        """Geometry fly-off / collapse only (not velocity or GRMS alone)."""
        return self.geometry_level == LEVEL_BAD


@dataclass(frozen=True)
class MonomerHealthReport:
    entries: tuple[MonomerHealthEntry, ...]
    flagged_bad: tuple[int, ...]
    flagged_warn: tuple[int, ...]
    baseline_recorded: bool
    restored: bool = False


@dataclass(frozen=True)
class MonomerHealthIntervention:
    """Result of :func:`maybe_intervene_monomer_health`.

    ``geometry_restored`` arms the full overlap rescue / READYN chain.
    ``velocities_redrawn`` only keeps Maxwell–Boltzmann velocities in RAM for
    the next overlap chunk (no MLpot SD, no truncated ``write restart``).
    """

    geometry_restored: bool = False
    velocities_redrawn: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.geometry_restored or self.velocities_redrawn)

    def __bool__(self) -> bool:
        return self.changed


def _entry_grms_for_selection(entry: MonomerHealthEntry) -> float:
    """Highest live GRMS signal for prioritizing limited intervention slots."""
    vals = [
        v
        for v in (entry.hybrid_grms_kcalmol_A, entry.charmm_grms_kcalmol_A)
        if v is not None and np.isfinite(v)
    ]
    if vals:
        return float(max(vals))
    return float("-inf")


def select_flagged_bad_by_highest_grms(
    report: MonomerHealthReport,
    *,
    max_select: int,
) -> tuple[int, ...]:
    """Select bad monomers by highest GRMS, not by residue index order."""
    if not report.flagged_bad:
        return ()
    entries_by_index = {int(entry.index): entry for entry in report.entries}
    ranked = sorted(
        (int(i) for i in report.flagged_bad),
        key=lambda i: (
            _entry_grms_for_selection(entries_by_index[i]),
            _level_rank(entries_by_index[i].overall_level),
            -i,
        )
        if i in entries_by_index
        else (float("-inf"), -1, -i),
        reverse=True,
    )
    return tuple(ranked[: max(1, int(max_select))])


def select_systemic_velocity_warn_by_highest_grms(
    report: MonomerHealthReport,
    *,
    min_fraction: float,
) -> tuple[int, ...]:
    """Select velocity-warn monomers when warnings are system-wide enough to recover."""
    if not report.entries or not report.flagged_warn:
        return ()
    velocity_warn = [
        entry
        for entry in report.entries
        if entry.velocity_level == LEVEL_WARN
        and entry.force_level != LEVEL_BAD
        and entry.geometry_level != LEVEL_BAD
    ]
    if not velocity_warn:
        return ()
    fraction = len(velocity_warn) / max(1, len(report.entries))
    if fraction < max(0.0, min(1.0, float(min_fraction))):
        return ()
    ranked = sorted(
        velocity_warn,
        key=lambda entry: (_entry_grms_for_selection(entry), -int(entry.index)),
        reverse=True,
    )
    return tuple(int(entry.index) for entry in ranked)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
        velocity_restore_on_template=not bool(
            getattr(args, "no_dynamics_monomer_velocity_restore", False)
        ),
        max_restore_per_check=max(
            1,
            _safe_int(getattr(args, "dynamics_monomer_health_max_restore", 4), 4),
        ),
        velocity_warn_ratio=_safe_float(
            getattr(args, "dynamics_monomer_velocity_warn_ratio", 3.0), 3.0
        ),
        velocity_bad_ratio=_safe_float(
            getattr(args, "dynamics_monomer_velocity_bad_ratio", 6.0), 6.0
        ),
        velocity_warn_abs_akma=_safe_float(
            getattr(args, "dynamics_monomer_velocity_warn_akma", 5000.0), 5000.0
        ),
        velocity_bad_abs_akma=_safe_float(
            getattr(args, "dynamics_monomer_velocity_bad_akma", 15000.0), 15000.0
        ),
        velocity_warn_recover_fraction=_safe_float(
            getattr(args, "dynamics_monomer_velocity_warn_recover_fraction", 0.80),
            0.80,
        ),
        force_warn_ratio=_safe_float(
            getattr(args, "dynamics_monomer_force_warn_ratio", 2.5), 2.5
        ),
        force_bad_ratio=_safe_float(
            getattr(args, "dynamics_monomer_force_bad_ratio", 5.0), 5.0
        ),
        energy_warn_ratio=_safe_float(
            getattr(args, "dynamics_monomer_energy_warn_ratio", 2.0), 2.0
        ),
        energy_bad_ratio=_safe_float(
            getattr(args, "dynamics_monomer_energy_bad_ratio", 4.0), 4.0
        ),
        baseline_floor_fraction_of_warn=_safe_float(
            getattr(args, "dynamics_monomer_baseline_floor_fraction", 0.25), 0.25
        ),
        ratio_requires_abs_warn=not bool(
            getattr(args, "dynamics_monomer_ratio_without_abs", False)
        ),
        template_restore_requires_geometry=not bool(
            getattr(args, "dynamics_monomer_template_on_force", False)
        ),
        bond_stretch_factor=_safe_float(
            getattr(args, "dynamics_monomer_bond_stretch_factor", 1.75), 1.75
        ),
        bond_stretch_abs_A=_safe_float(
            getattr(args, "dynamics_monomer_bond_stretch_abs", 2.50), 2.50
        ),
        com_flyoff_enabled=not bool(
            getattr(args, "no_dynamics_monomer_com_flyoff", False)
        ),
        com_flyoff_A=_safe_float(
            getattr(args, "dynamics_monomer_com_flyoff", 0.0), 0.0
        ),
        verbose=not bool(getattr(args, "quiet", False)),
    )


def resolve_monomer_offsets_for_ctx(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    n_atoms: int,
) -> np.ndarray | None:
    """Return cumulative monomer offsets, preferring PSF / composition atom counts."""
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
    # Mixed systems (MEOH + TIP3, …) when live PSF resid parsing is unavailable.
    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is not None:
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            _cluster_atoms_per_from_composition,
        )

        comp_per = _cluster_atoms_per_from_composition(args, n_atoms=int(n_atoms))
        if (
            comp_per is not None
            and len(comp_per) == int(n_monomers)
            and int(sum(comp_per)) == int(n_atoms)
        ):
            return monomer_offsets_from_atoms_per(comp_per)
    if int(n_monomers) > 0 and int(n_atoms) > 0 and int(n_atoms) % int(n_monomers) == 0:
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
    baseline_floor: float = 0.0,
    ratio_requires_abs_warn: bool = True,
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
        level = LEVEL_WARN
        reasons.append(f"{name} abs {val:.1f} ≥ {warn_abs:.1f}")
    # Ratios never promote alone: require abs floor, and floor a tiny baseline.
    if baseline is not None and np.isfinite(baseline):
        base = max(float(baseline), float(baseline_floor), 1.0e-8)
        if (not ratio_requires_abs_warn) or val >= float(warn_abs):
            ratio = val / base
            if ratio >= float(bad_ratio) and val >= float(bad_abs):
                level = LEVEL_BAD
                reasons.append(f"{name} ratio {ratio:.1f}× baseline")
            elif ratio >= float(warn_ratio) and val >= float(warn_abs):
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
    if int(n_monomers) <= 1:
        return None
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    n_atoms = int(coor.get_natom())
    if n_atoms <= 0:
        return None
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
    # Re-seed unwrapped COM baseline whenever health metrics baseline resets.
    setattr(mlpot_ctx, "_monomer_com_unwrap_reset", True)
    return baseline


def _get_baseline(mlpot_ctx: Any) -> MonomerHealthBaseline | None:
    baseline = getattr(mlpot_ctx, "_monomer_health_baseline", None)
    if isinstance(baseline, MonomerHealthBaseline):
        return baseline
    return None


def _monomer_coms_numpy(
    positions: np.ndarray,
    offsets: np.ndarray,
    masses: np.ndarray | None = None,
) -> np.ndarray:
    """Geometric or mass-weighted COM per monomer, shape ``(n_monomers, 3)``."""
    pos = np.asarray(positions, dtype=np.float64)
    off = np.asarray(offsets, dtype=int)
    n_mon = max(0, int(len(off) - 1))
    coms = np.zeros((n_mon, 3), dtype=np.float64)
    m = None if masses is None else np.asarray(masses, dtype=np.float64).reshape(-1)
    for mi in range(n_mon):
        s, e = int(off[mi]), int(off[mi + 1])
        if e <= s:
            continue
        chunk = pos[s:e]
        if m is not None and m.shape[0] >= e:
            w = np.maximum(m[s:e], 1.0e-12)
            coms[mi] = (chunk * w[:, None]).sum(axis=0) / w.sum()
        else:
            coms[mi] = chunk.mean(axis=0)
    return coms


def _mic_delta(a: np.ndarray, b: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    """Minimum-image ``b - a`` (Å)."""
    from mmml.utils.geometry_checks import _cell_matrix, _mic

    cell_mat = _cell_matrix(cell)
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return _mic(d.reshape(-1, 3), cell_mat).reshape(d.shape)


def _resolve_com_flyoff_threshold_A(
    config: MonomerHealthConfig,
    overlap_config: Any,
    cell: Any | None,
) -> float:
    explicit = float(getattr(config, "com_flyoff_A", 0.0) or 0.0)
    if explicit > 0.0:
        return explicit
    side = float(getattr(overlap_config, "fallback_box_side_A", 0.0) or 0.0)
    if side <= 0.0 and cell is not None:
        from mmml.utils.geometry_checks import _cell_matrix

        mat = _cell_matrix(cell)
        if mat is not None:
            side = float(np.mean(np.linalg.norm(mat, axis=1)))
    if side > 0.0:
        return max(15.0, 0.35 * side)
    return 15.0


def _update_com_unwrap_state(
    mlpot_ctx: Any,
    coms_wrapped: np.ndarray,
    cell: Any | None,
    *,
    reset_baseline: bool = False,
) -> np.ndarray:
    """Return unwrapped COMs; seed / refresh state on ``mlpot_ctx``."""
    wrapped = np.asarray(coms_wrapped, dtype=np.float64)
    state = getattr(mlpot_ctx, "_monomer_com_unwrap_state", None)
    if (
        reset_baseline
        or not isinstance(state, dict)
        or state.get("last_wrapped") is None
        or np.asarray(state["last_wrapped"]).shape != wrapped.shape
    ):
        state = {
            "last_wrapped": wrapped.copy(),
            "unwrapped": wrapped.copy(),
            "baseline_unwrapped": wrapped.copy(),
        }
        setattr(mlpot_ctx, "_monomer_com_unwrap_state", state)
        setattr(mlpot_ctx, "_monomer_com_unwrap_reset", False)
        return wrapped.copy()

    last = np.asarray(state["last_wrapped"], dtype=np.float64)
    unwrapped = np.asarray(state["unwrapped"], dtype=np.float64).copy()
    for mi in range(wrapped.shape[0]):
        unwrapped[mi] = unwrapped[mi] + _mic_delta(last[mi], wrapped[mi], cell)
    state["last_wrapped"] = wrapped.copy()
    state["unwrapped"] = unwrapped
    setattr(mlpot_ctx, "_monomer_com_unwrap_state", state)
    return unwrapped


def _flag_bond_stretch_monomers(
    positions: np.ndarray,
    offsets: np.ndarray,
    *,
    stretch_factor: float,
    stretch_abs_A: float,
    ref_positions: np.ndarray | None,
) -> dict[int, tuple[str, ...]]:
    """Flag monomers with overstretched PSF 1–2 bonds."""
    out: dict[int, tuple[str, ...]] = {}
    if float(stretch_factor) <= 1.0 and float(stretch_abs_A) <= 0.0:
        return out
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.monomer_geometry_limits import (
            psf_bond_pairs_0based,
        )

        bonds = psf_bond_pairs_0based(exclude_1_3=False)
    except Exception:
        return out
    if not bonds:
        return out

    pos = np.asarray(positions, dtype=np.float64)
    ref = None if ref_positions is None else np.asarray(ref_positions, dtype=np.float64)
    off = np.asarray(offsets, dtype=int)
    n_mon = max(0, int(len(off) - 1))
    for mi in range(n_mon):
        s, e = int(off[mi]), int(off[mi + 1])
        worst_ratio = 0.0
        worst_len = 0.0
        worst_ref = 0.0
        for a, b in bonds:
            if not (s <= a < e and s <= b < e):
                continue
            length = float(np.linalg.norm(pos[b] - pos[a]))
            ref_len = None
            if ref is not None and ref.shape == pos.shape:
                ref_len = float(np.linalg.norm(ref[b] - ref[a]))
            limit = float(stretch_abs_A)
            if ref_len is not None and ref_len > 0.05:
                limit = max(limit, float(stretch_factor) * ref_len)
            if length > limit and length > worst_len:
                worst_len = length
                worst_ref = float(ref_len) if ref_len is not None else 0.0
                worst_ratio = length / max(ref_len, 1.0e-8) if ref_len else 0.0
        if worst_len > 0.0:
            if worst_ref > 0.0:
                out[mi] = (
                    f"bond {worst_len:.2f} Å > limit "
                    f"({float(stretch_factor):.2f}×{worst_ref:.2f} / "
                    f"abs {float(stretch_abs_A):.2f} Å; {worst_ratio:.1f}×)",
                )
            else:
                out[mi] = (
                    f"bond {worst_len:.2f} Å > abs {float(stretch_abs_A):.2f} Å",
                )
    return out


def flag_geometry_problem_monomers(
    mlpot_ctx: Any,
    overlap_config: Any,
    *,
    offsets: np.ndarray,
    health_config: MonomerHealthConfig | None = None,
) -> dict[int, tuple[str, ...]]:
    """Return monomers with fly-off / bond-stretch / intra-collapse geometry failures.

    Extent catches torn monomers; bond stretch catches early dissociation; COM
    unwrap drift catches rigid-body escapes even when IMAGE centering rewraps
    coordinates into the primary cell.
    """
    from mmml.interfaces.pycharmmInterface.mlpot.overlap_guard import (
        _bond_exclusion_pairs,
        _overlap_cell,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array
    from mmml.utils.geometry_checks import monomer_axis_extent

    cfg = health_config or MonomerHealthConfig()
    out: dict[int, tuple[str, ...]] = {}
    max_extent = float(getattr(overlap_config, "max_monomer_extent_A", 0.0) or 0.0)
    intra_min = float(getattr(overlap_config, "intra_min_distance_A", 0.0) or 0.0)

    pos = get_charmm_positions_array()
    cell = _overlap_cell(
        use_pbc=bool(getattr(overlap_config, "use_pbc", False)),
        fallback_box_side_A=getattr(overlap_config, "fallback_box_side_A", None),
    )
    n_monomers = max(0, int(len(offsets) - 1))
    if max_extent > 0.0:
        for mi in range(n_monomers):
            extent = float(monomer_axis_extent(pos, offsets, mi, cell=cell))
            if extent > max_extent:
                out[mi] = out.get(mi, ()) + (
                    f"extent {extent:.2f} Å > {max_extent:.2f} Å",
                )

    ref_pos = None
    for attr in ("geometry_baseline_positions", "geometry_mini_positions"):
        cand = getattr(mlpot_ctx, attr, None)
        if cand is None:
            continue
        arr = np.asarray(cand, dtype=np.float64)
        if arr.shape == np.asarray(pos).shape and np.all(np.isfinite(arr)):
            ref_pos = arr
            break

    for mi, reasons in _flag_bond_stretch_monomers(
        pos,
        offsets,
        stretch_factor=float(cfg.bond_stretch_factor),
        stretch_abs_A=float(cfg.bond_stretch_abs_A),
        ref_positions=ref_pos,
    ).items():
        out[mi] = out.get(mi, ()) + reasons

    if bool(cfg.com_flyoff_enabled) and n_monomers > 1:
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
                charmm_masses_amu,
            )

            masses = charmm_masses_amu()
        except Exception:
            masses = None
        coms = _monomer_coms_numpy(pos, offsets, masses=masses)
        reset = bool(getattr(mlpot_ctx, "_monomer_com_unwrap_reset", False))
        if reset:
            setattr(mlpot_ctx, "_monomer_com_unwrap_reset", False)
        unwrapped = _update_com_unwrap_state(
            mlpot_ctx, coms, cell, reset_baseline=reset
        )
        state = getattr(mlpot_ctx, "_monomer_com_unwrap_state", None)
        baseline = (
            np.asarray(state["baseline_unwrapped"], dtype=np.float64)
            if isinstance(state, dict) and state.get("baseline_unwrapped") is not None
            else unwrapped
        )
        thresh = _resolve_com_flyoff_threshold_A(cfg, overlap_config, cell)
        for mi in range(n_monomers):
            drift = float(np.linalg.norm(unwrapped[mi] - baseline[mi]))
            if drift > thresh:
                out[mi] = out.get(mi, ()) + (
                    f"COM drift {drift:.1f} Å > {thresh:.1f} Å (unwrapped)",
                )

    if intra_min > 0.0:
        from mmml.utils.geometry_checks import find_worst_intramonomer_close_contact

        excluded = _bond_exclusion_pairs(
            exclude_1_3=bool(getattr(overlap_config, "intra_exclude_1_3", True))
        )
        marked: set[int] = set()
        work_pos = np.asarray(pos, dtype=np.float64)
        for _ in range(max(1, n_monomers * 2)):
            dist, viol = find_worst_intramonomer_close_contact(
                work_pos,
                offsets,
                excluded,
                cell=cell,
                min_distance=intra_min,
            )
            if viol is None or float(dist) >= intra_min:
                break
            mi = int(viol.monomer)
            if mi not in marked:
                marked.add(mi)
                out[mi] = out.get(mi, ()) + (
                    f"intra {float(dist):.3f} Å < {intra_min:.3f} Å",
                )
            mid = 0.5 * (work_pos[int(viol.atom_i)] + work_pos[int(viol.atom_j)])
            work_pos[int(viol.atom_i)] = mid
            work_pos[int(viol.atom_j)] = mid
    return out


def audit_monomer_health(
    mlpot_ctx: Any,
    config: MonomerHealthConfig,
    *,
    n_monomers: int,
    global_step: int | None = None,
    overlap_config: Any | None = None,
) -> MonomerHealthReport | None:
    """Classify per-monomer velocity / GRMS / geometry stress vs baseline."""
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
    geom_reasons = {}
    if overlap_config is not None:
        geom_reasons = flag_geometry_problem_monomers(
            mlpot_ctx,
            overlap_config,
            offsets=offsets,
            health_config=config,
        )

    v_floor = float(config.velocity_warn_abs_akma) * float(
        config.baseline_floor_fraction_of_warn
    )
    f_floor = float(config.force_warn_abs_kcalmol_A) * float(
        config.baseline_floor_fraction_of_warn
    )
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
            baseline_floor=v_floor,
            ratio_requires_abs_warn=bool(config.ratio_requires_abs_warn),
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
            baseline_floor=f_floor,
            ratio_requires_abs_warn=bool(config.ratio_requires_abs_warn),
        )
        g_reasons = geom_reasons.get(int(mi), ())
        g_level = LEVEL_BAD if g_reasons else LEVEL_OK
        # energy_level kept for matrix layout; store geometry status there too.
        reasons = v_reasons + f_reasons + g_reasons
        entry = MonomerHealthEntry(
            index=int(mi),
            label=str(labels[mi]) if mi < len(labels) else f"M{mi:02d}",
            velocity_rms_akma=float(vel_rms[mi]) if np.isfinite(vel_rms[mi]) else None,
            velocity_max_akma=float(vel_max[mi]) if np.isfinite(vel_max[mi]) else None,
            hybrid_grms_kcalmol_A=float(hybrid[mi]) if np.isfinite(hybrid[mi]) else None,
            charmm_grms_kcalmol_A=float(charmm[mi]) if np.isfinite(charmm[mi]) else None,
            velocity_level=v_level,
            force_level=f_level,
            energy_level=g_level,
            reasons=reasons,
            geometry_level=g_level,
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
    """Print residue grid: green/yellow/red dots for velocity, GRMS, geometry."""
    if not report.entries:
        return
    from mmml.utils.rich_report import emit, rich_enabled

    use_rich = rich_enabled(quiet=quiet)
    header = (
        "Monomer health  [v]=velocity [f]=GRMS "
        "[g]=geometry (extent/bond/COM-drift/intra)"
    )
    lines = [header, " idx   v f g  residue"]
    for entry in report.entries:
        g_level = getattr(entry, "geometry_level", entry.energy_level)
        if use_rich:
            v_dot = _DOT_RICH[entry.velocity_level]
            f_dot = _DOT_RICH[entry.force_level]
            g_dot = _DOT_RICH[g_level]
        else:
            v_dot = _DOT_PLAIN[entry.velocity_level]
            f_dot = _DOT_PLAIN[entry.force_level]
            g_dot = _DOT_PLAIN[g_level]
        dots = f"{v_dot} {f_dot} {g_dot}"
        lines.append(
            f" {entry.index:3d}  {dots}  {entry.label}"
            + (f"  ({'; '.join(entry.reasons)})" if entry.reasons else "")
        )
    body = "\n".join(lines)
    if use_rich:
        emit(f"[bold]{context}[/bold]\n{body}", quiet=quiet)
    else:
        emit(f"{context}\n{body}", quiet=quiet)


def _resolve_health_velocity_temperature_K(mlpot_ctx: Any) -> float:
    """Target bath temperature for per-monomer velocity redraw after template restore."""
    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is None:
        return 300.0
    for attr in ("heat_finalt", "heat_firstt", "temperature", "temp"):
        raw = getattr(args, attr, None)
        if raw is None:
            continue
        try:
            temp = float(raw)
        except (TypeError, ValueError):
            continue
        if temp > 0.0:
            return temp
    return 300.0


def _current_velocities_akma(n_atoms: int) -> np.ndarray | None:
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
            charmm_synced_velocities_akma,
        )

        vel = charmm_synced_velocities_akma()
        if vel is not None and int(vel.shape[0]) >= int(n_atoms):
            return np.asarray(vel[:n_atoms], dtype=np.float64).copy()
    except Exception:
        pass
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
            _charmm_velocities_array,
        )

        vel = _charmm_velocities_array()
        if vel is not None and int(vel.shape[0]) >= int(n_atoms):
            return np.asarray(vel[:n_atoms], dtype=np.float64).copy()
    except Exception:
        pass
    return None


def restore_monomer_velocities_from_template(
    mlpot_ctx: Any,
    flagged: tuple[int, ...] | list[int],
    *,
    offsets: np.ndarray,
    template_source: Path | str,
    temperature_K: float | None = None,
    verbose: bool = False,
) -> bool:
    """Splice template restart velocities onto flagged monomers (or MB redraw)."""
    if not flagged:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _maxwell_boltzmann_akma_numpy,
        _read_restart_velocities_akma,
        charmm_masses_amu,
        sync_charmm_velocities_akma,
        velocities_are_pathological,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_velocities,
    )

    n_atoms = int(offsets[-1])
    vel = _current_velocities_akma(n_atoms)
    if vel is None:
        vel = np.zeros((n_atoms, 3), dtype=np.float64)

    source = Path(template_source)
    ref_vel = read_restart_velocities(source)
    if ref_vel is None:
        ref_vel = _read_restart_velocities_akma(source, quiet=True)
    if ref_vel is not None:
        ref_vel = np.asarray(ref_vel, dtype=np.float64).reshape(-1, 3)

    masses = charmm_masses_amu()
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        clamp_velocity_assignment_temp_k,
    )

    temp = clamp_velocity_assignment_temp_k(
        float(
            temperature_K
            if temperature_K is not None
            else _resolve_health_velocity_temperature_K(mlpot_ctx)
        )
    )
    modified = False
    for mi in flagged:
        start = int(offsets[int(mi)])
        end = int(offsets[int(mi) + 1])
        if end <= start:
            continue
        if (
            ref_vel is not None
            and ref_vel.shape[0] >= end
            and not velocities_are_pathological(
                ref_vel[start:end],
                masses_amu=masses[start:end],
            )
        ):
            vel[start:end] = ref_vel[start:end]
            modified = True
            continue
        vel[start:end] = _maxwell_boltzmann_akma_numpy(masses[start:end], temp)
        modified = True

    if not modified:
        return False
    sync_charmm_velocities_akma(vel)
    if verbose:
        print(
            f"Template velocity restore: monomer(s) {list(flagged)} "
            f"from {source.name} (fallback T={temp:.1f} K where needed)",
            flush=True,
        )
    return True


def redraw_monomer_velocities(
    mlpot_ctx: Any,
    flagged: tuple[int, ...] | list[int],
    *,
    offsets: np.ndarray,
    temperature_K: float | None = None,
    verbose: bool = False,
    context: str = "monomer health",
) -> bool:
    """Maxwell-Boltzmann redraw for selected monomers without moving coordinates."""
    if not flagged:
        return False
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _maxwell_boltzmann_akma_numpy,
        charmm_masses_amu,
        sync_charmm_velocities_akma,
    )

    n_atoms = int(offsets[-1])
    vel = _current_velocities_akma(n_atoms)
    if vel is None:
        vel = np.zeros((n_atoms, 3), dtype=np.float64)
    masses = charmm_masses_amu()
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        clamp_velocity_assignment_temp_k,
    )

    temp = clamp_velocity_assignment_temp_k(
        float(
            temperature_K
            if temperature_K is not None
            else _resolve_health_velocity_temperature_K(mlpot_ctx)
        )
    )
    modified = False
    selected = [int(i) for i in flagged]
    for mi in selected:
        start = int(offsets[int(mi)])
        end = int(offsets[int(mi) + 1])
        if end <= start:
            continue
        vel[start:end] = _maxwell_boltzmann_akma_numpy(masses[start:end], temp)
        modified = True
    if not modified:
        return False
    sync_charmm_velocities_akma(vel)
    if verbose:
        print(
            f"{context}: redrew velocities for monomer(s) {selected} "
            f"(T={temp:.1f} K)",
            flush=True,
        )
    return True


def restore_flagged_monomers_from_template(
    mlpot_ctx: Any,
    flagged: tuple[int, ...] | list[int],
    *,
    context: str,
    restart_path: Any | None = None,
    verbose: bool = False,
    velocity_restore: bool = True,
    temperature_K: float | None = None,
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
    if velocity_restore:
        restore_monomer_velocities_from_template(
            mlpot_ctx,
            selected,
            offsets=offsets,
            template_source=source,
            temperature_K=temperature_K,
            verbose=verbose,
        )
    return True


def _run_per_monomer_jax_on_indices(
    mlpot_ctx: Any,
    overlap_config: Any,
    monomer_indices: tuple[int, ...],
    *,
    context: str,
    restart_path: Any | None = None,
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
    grms = minimize_bonded_jax_per_monomer_recovery(
        mlpot_ctx,
        bonded_cfg,
        n_monomers=int(getattr(overlap_config, "n_monomers", 1) or 1),
        topology_psf=topo,
        context=f"{context} (per-monomer JAX)",
        monomer_indices=monomer_indices,
    )
    if grms is not None:
        return

    from mmml.interfaces.pycharmmInterface.mlpot.monomer_physnet_mini import (
        monomer_physnet_mini_enabled,
        run_selective_monomer_physnet_mini,
        selective_monomer_physnet_mini_config_from_args,
    )

    args = getattr(mlpot_ctx, "workflow_args", None)
    if not monomer_physnet_mini_enabled(args):
        return
    physnet_cfg = selective_monomer_physnet_mini_config_from_args(
        args,
        verbose=bool(getattr(bonded_cfg, "verbose", False)),
        quiet_bfgs=bool(getattr(args, "quiet_bfgs", False)) if args is not None else False,
    )
    run_selective_monomer_physnet_mini(
        mlpot_ctx,
        config=physnet_cfg,
        context_prefix=f"{context} monomer PhysNet",
        flagged=monomer_indices,
        restart_path=restart_path,
    )


def maybe_rebaseline_monomer_health_after_heat_velocities(
    mlpot_ctx: Any,
    *,
    n_monomers: int,
    context: str,
    global_step: int,
) -> bool:
    """Re-record baseline once after HEAT has assigned thermal velocities.

    Initial baseline is often post-mini (near-zero GRMS / cold). Comparing mid-heat
    metrics to that reference produces misleading hundreds× ratios.
    """
    if mlpot_ctx is None or int(n_monomers) <= 1:
        return False
    if "heat" not in str(context).lower():
        return False
    if int(global_step) <= 0:
        return False
    if bool(getattr(mlpot_ctx, "_monomer_health_rebaselined_after_heat_vel", False)):
        return False
    record_monomer_health_baseline(
        mlpot_ctx,
        n_monomers=int(n_monomers),
        global_step=int(global_step),
    )
    setattr(mlpot_ctx, "_monomer_health_rebaselined_after_heat_vel", True)
    return True


def _velocity_only_redraw_indices(report: MonomerHealthReport) -> tuple[int, ...]:
    """Monomers with hot velocity but healthy geometry (no template restore)."""
    out: list[int] = []
    for entry in report.entries:
        if entry.geometry_level == LEVEL_BAD:
            continue
        if entry.velocity_level == LEVEL_BAD:
            out.append(int(entry.index))
    return tuple(out)


def maybe_intervene_monomer_health(
    mlpot_ctx: Any,
    overlap_config: Any,
    *,
    context: str,
    global_step: int | None = None,
    restart_path: Any | None = None,
) -> MonomerHealthIntervention:
    """Audit health; template+FIRE only for geometry; redraw hot velocities otherwise.

    Only ``geometry_restored`` should enter the overlap MLpot-SD / READYN rescue chain.
    Velocity redraws keep state in RAM for the next chunk.
    """
    health_cfg = getattr(overlap_config, "monomer_health", None)
    if health_cfg is None:
        args = getattr(mlpot_ctx, "workflow_args", None)
        health_cfg = monomer_health_config_from_args(args)
    if not health_cfg.enabled:
        return MonomerHealthIntervention()

    n_monomers = int(getattr(overlap_config, "n_monomers", 1) or 1)
    if global_step is not None:
        maybe_rebaseline_monomer_health_after_heat_velocities(
            mlpot_ctx,
            n_monomers=n_monomers,
            context=context,
            global_step=int(global_step),
        )

    report = audit_monomer_health(
        mlpot_ctx,
        health_cfg,
        n_monomers=n_monomers,
        global_step=global_step,
        overlap_config=overlap_config,
    )
    if report is None or report.baseline_recorded:
        return MonomerHealthIntervention()

    if health_cfg.debug_dot_matrix or report.flagged_bad or report.flagged_warn:
        emit_monomer_health_dot_matrix(
            report,
            context=f"{context} monomer health (step {global_step})",
            quiet=not health_cfg.verbose and not health_cfg.debug_dot_matrix,
        )

    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm.coor as coor

    n_atoms = int(coor.get_natom())
    offsets = resolve_monomer_offsets_for_ctx(
        mlpot_ctx, n_monomers=int(n_monomers), n_atoms=n_atoms
    )
    if offsets is None:
        return MonomerHealthIntervention()

    geometry_restored = False
    velocities_redrawn = False

    geometry_bad = tuple(
        int(e.index) for e in report.entries if e.geometry_level == LEVEL_BAD
    )
    if geometry_bad and health_cfg.template_restore_on_bad:
        if bool(health_cfg.template_restore_requires_geometry):
            to_restore = select_flagged_bad_by_highest_grms(
                MonomerHealthReport(
                    entries=report.entries,
                    flagged_bad=geometry_bad,
                    flagged_warn=(),
                    baseline_recorded=False,
                ),
                max_select=int(health_cfg.max_restore_per_check),
            )
        else:
            to_restore = select_flagged_bad_by_highest_grms(
                report,
                max_select=int(health_cfg.max_restore_per_check),
            )
        if to_restore:
            restored = restore_flagged_monomers_from_template(
                mlpot_ctx,
                to_restore,
                context=context,
                restart_path=restart_path,
                verbose=health_cfg.verbose or health_cfg.debug_dot_matrix,
                velocity_restore=bool(health_cfg.velocity_restore_on_template),
                temperature_K=_resolve_health_velocity_temperature_K(mlpot_ctx),
            )
            if restored:
                geometry_restored = True
                if health_cfg.per_monomer_jax_after_restore:
                    _run_per_monomer_jax_on_indices(
                        mlpot_ctx,
                        overlap_config,
                        tuple(to_restore),
                        context=context,
                        restart_path=restart_path,
                    )
                mlpot_ctx.reregister_mlpot(verbose=False)
    elif geometry_bad and health_cfg.verbose:
        print(
            f"{context}: geometry-bad monomers {geometry_bad} "
            "(template restore disabled)",
            flush=True,
        )

    # Hot velocities without a geometry failure: redraw only (no template / FIRE).
    to_redraw = _velocity_only_redraw_indices(report)
    if not to_redraw:
        to_redraw = select_systemic_velocity_warn_by_highest_grms(
            report,
            min_fraction=float(health_cfg.velocity_warn_recover_fraction),
        )
    # Do not redraw monomers we just template-restored.
    if geometry_bad and health_cfg.template_restore_on_bad:
        geom_set = set(geometry_bad)
        to_redraw = tuple(i for i in to_redraw if i not in geom_set)
    if to_redraw:
        redrawn = redraw_monomer_velocities(
            mlpot_ctx,
            to_redraw,
            offsets=offsets,
            temperature_K=_resolve_health_velocity_temperature_K(mlpot_ctx),
            verbose=health_cfg.verbose or health_cfg.debug_dot_matrix,
            context=context,
        )
        velocities_redrawn = bool(redrawn)

    result = MonomerHealthIntervention(
        geometry_restored=geometry_restored,
        velocities_redrawn=velocities_redrawn,
    )
    if result.changed:
        from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
            invalidate_mlpot_calculator_caches,
        )

        invalidate_mlpot_calculator_caches(mlpot_ctx)
    return result
