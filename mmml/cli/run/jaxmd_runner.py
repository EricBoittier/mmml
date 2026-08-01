"""JAX-MD simulation setup and Nose-Hoover chain routines."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax_md
import numpy as np
from jax import grad, jit, lax
from jax_md import quantity, simulate, space, units
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import jax.numpy as jnp

from mmml.cli.run.summaries import (
    print_flat_bottom_summary,
    print_forces_summary,
    save_calculator_summary_json,
)
from mmml.utils.rich_report import emit_md_system_calculator_report
from mmml.interfaces.pycharmmInterface.pbc_utils_jax import (
    group_ids_from_groups,
    wrap_groups_by_id_with_weight_sum,
)
from mmml.interfaces.pycharmmInterface.ml_dtypes import resolve_ml_compute_dtype
from mmml.utils.geometry_checks import (
    TEMPLATE_DONOR_IDEAL_TIP3,
    TEMPLATE_DONOR_MAX_FORCE_EVA,
    assert_no_intermonomer_atom_overlap,
    rebuild_high_force_monomers_from_peers,
)
from mmml.utils.hdf5_reporter import make_jaxmd_reporter
from mmml.utils.jax_gpu_warmup import block_jax_values, ensure_xla_gpu_warmed

import ase.io as ase_io
from typing import Callable, Optional


def directional_force_energy_error(
    energy_plus: float,
    energy_minus: float,
    epsilon_A: float,
    projected_force_eV_A: float,
) -> tuple[float, float]:
    """Compare a finite-difference energy slope with ``-F·direction``."""
    eps = float(epsilon_A)
    if eps <= 0.0:
        raise ValueError("epsilon_A must be positive")
    derivative = (float(energy_plus) - float(energy_minus)) / (2.0 * eps)
    expected = -float(projected_force_eV_A)
    scale = max(abs(derivative), abs(expected), 1.0e-3)
    return derivative, abs(derivative - expected) / scale


def nve_force_energy_ablation_verdict(
    hybrid_relerr: float,
    ml_only_relerr: float | None,
    tol: float,
    *,
    mm_charge_mode: str | None = None,
    used_frozen_mm_charges: bool = False,
) -> str:
    """Interpret hybrid vs ML-only (doMM=False) directional FD relative errors.

    Used when NVE preflight fails to separate PBC ML-dimer / switch issues from
    MM pair-list / E_MM assembly.
    """
    hyb = float(hybrid_relerr)
    tol = float(tol)
    if ml_only_relerr is None or not np.isfinite(ml_only_relerr):
        return "ML-only ablation unavailable"
    ml = float(ml_only_relerr)
    hyb_fail = (not np.isfinite(hyb)) or hyb > tol
    ml_fail = (not np.isfinite(ml)) or ml > tol
    if not hyb_fail and not ml_fail:
        return "hybrid and ML-only both within tolerance"
    if hyb_fail and not ml_fail:
        mode = (mm_charge_mode or "").strip().lower()
        if mode in {"q0", "latent", "q1", "fixed_plus_latent", "latent_dynamic"} and (
            not used_frozen_mm_charges
        ):
            return (
                "ML-only passes → likely Hellmann–Feynman mismatch: E_MM uses "
                f"position-dependent MM charges (mm_charge_mode={mode}) while MM "
                "forces hold q fixed (train-matched). Re-run with frozen-q NVE "
                "preflight (default for these modes) or use --mm-charge-mode fixed"
            )
        return (
            "ML-only passes → suspect MM / hybrid assembly (pairs, E_MM, handoff mix); "
            "PBC ML-dimer path looks conservative"
        )
    if hyb_fail and ml_fail:
        # Similar magnitude → same root; much larger hybrid → MM adds damage.
        if np.isfinite(hyb) and np.isfinite(ml) and hyb > 1.5 * max(ml, 1.0e-12) and hyb > tol:
            return (
                "both fail, hybrid worse → MM/hybrid assembly adds non-conservatism on "
                "top of an already-failing ML path"
            )
        return (
            "ML-only also fails → suspect PBC ML-dimer / MIC wrap / switch forces "
            "(not MM pairs)"
        )
    return (
        "ML-only FD fails but hybrid gate passed — continuing "
        "(ML-only is diagnostic; NVE uses hybrid forces)"
    )


def nve_force_energy_should_attempt_rescue(
    hybrid_relerr: float,
    tol: float,
    *,
    rescue_enabled: bool,
    rescue_already_attempted: bool,
) -> bool:
    """True when hybrid FD failed and one NL+FIRE rescue cycle is still available."""
    if not rescue_enabled or rescue_already_attempted:
        return False
    if not np.isfinite(hybrid_relerr):
        return True
    return float(hybrid_relerr) > float(tol)


# Post-FIRE NVE start max|F| gate.  The historical 1.5 eV/Å ceiling was tuned on
# small clusters; dense liquids (TIP3:903, N≈2700) routinely finish FIRE near
# 5–7 eV/Å while still passing force–energy FD.  Scale the configured base with
# sqrt(N / N_ref) so large systems get a higher ceiling without disabling the gate.
NVE_MAX_F_START_BASE_EVA = 1.5
NVE_MAX_F_START_N_ATOMS_REF = 100
NVE_MAX_F_START_SCALE_EXP = 0.5
NVE_MAX_F_START_MAX_EVA = 15.0


def resolve_nve_max_f_start_gate_eVA(
    configured_eVA: float | None,
    *,
    n_atoms: int,
    n_atoms_ref: int = NVE_MAX_F_START_N_ATOMS_REF,
    scale_exp: float = NVE_MAX_F_START_SCALE_EXP,
    max_eVA: float = NVE_MAX_F_START_MAX_EVA,
) -> tuple[float, float]:
    """Return ``(effective_gate_eVA, size_scale)`` for the NVE start max|F| check.

    ``configured_eVA <= 0`` disables the gate (returns ``(0.0, 1.0)``).
    Otherwise ``effective = min(max_eVA, configured * max(1, (N / N_ref)^exp))``.
    """
    base = float(NVE_MAX_F_START_BASE_EVA if configured_eVA is None else configured_eVA)
    if base <= 0.0:
        return 0.0, 1.0
    n = max(int(n_atoms), 1)
    n_ref = max(int(n_atoms_ref), 1)
    scale = max(1.0, (float(n) / float(n_ref)) ** float(scale_exp))
    return float(min(float(max_eVA), base * scale)), float(scale)


def nve_etot_drift_should_attempt_rescue(
    *,
    rescue_enabled: bool,
    attempts_used: int,
    max_attempts: int,
) -> bool:
    """True when mid-run NVE E_tot drift still has rescue budget."""
    if not rescue_enabled:
        return False
    return int(attempts_used) < int(max_attempts)


def nve_etot_drift_rescue_tricks(attempt_index: int) -> tuple[str, ...]:
    """Ordered repair tricks for one E_tot-drift rescue attempt (0-based).

    Escalates from cheap NL/velocity fixes to CHARMM geometry, FIRE, and
    MD timestep backoff.  Grace widens the E_tot gate after every attempt.
    """
    i = int(attempt_index)
    if i <= 0:
        return ("nl_rebuild", "rethermalize", "grace")
    if i == 1:
        return ("nl_rebuild", "fire", "rethermalize", "dt_halve", "grace")
    # Attempt 2+: throw everything we have.
    return (
        "charmm_rescue",
        "nl_rebuild",
        "fire",
        "rethermalize",
        "dt_halve",
        "grace",
    )


def nve_etot_drift_grace_threshold_eV(
    *,
    current_threshold_eV: float,
    grace_eV: float,
    attempt_1_based: int,
) -> float:
    """Progressive E_tot gate after rescue attempt ``attempt_1_based`` (1, 2, …)."""
    g = float(grace_eV)
    if g <= 0.0:
        return float(current_threshold_eV)
    # attempt 1 → g, 2 → 1.5 g, 3 → 2 g, …
    widened = g * (1.0 + 0.5 * max(0, int(attempt_1_based) - 1))
    return max(float(current_threshold_eV), float(widened))


def nve_etot_drift_halved_dt_ps(
    dt_ps: float,
    *,
    scale: float = 0.5,
    min_dt_fs: float = 0.05,
) -> float:
    """Next MD timestep (ps) after a drift-rescue halve, floored at ``min_dt_fs``."""
    dt_new = float(dt_ps) * float(scale)
    min_ps = float(min_dt_fs) * 0.001
    return max(dt_new, min_ps)


def _nl_update_positions(positions):
    """Pass JAX arrays to ``update_mm_pairs`` so it can avoid host sync on cache hits."""
    import os

    if hasattr(positions, "__dlpack_device__") and os.environ.get("MMML_MM_NL_FORCE_HOST") != "1":
        return positions
    return np.asarray(positions)

WORSE_COUNT_THRESHOLD = 100
# jax-md FIRE timesteps are in the metal unit system (ps).  Historical default
# 1e-3 overshoots on hard liquid starts (max|F| ≳ 1 eV/Å); those use a colder ladder.
DEFAULT_JAXMD_FIRE_DT_PS = 1.0e-3
JAXMD_FIRE_DT_SOFT_PS = 1.0e-4
JAXMD_FIRE_DT_MILD_PS = 3.0e-4
JAXMD_FIRE_DT_HIGH_F_PS = 1.0e-4
JAXMD_FIRE_DT_VERY_HIGH_F_PS = 5.0e-5
JAXMD_FIRE_DT_MIN_PS = 1.0e-5
JAXMD_FIRE_NL_REFRESH_EVERY = 25
# ASE rescue target is typically 0.1 eV/Å. Below this, jax-md FIRE rarely helps
# and just burns stages on a flat landscape — skip unless explicitly forced.
DEFAULT_JAXMD_FIRE_SKIP_MAX_F_EVA = 0.10
# max|F| gates for ``resolve_jaxmd_fire_dt_start_ps`` (eV/Å).
JAXMD_FIRE_FMAX_SOFT_EVA = 0.15
JAXMD_FIRE_FMAX_MILD_EVA = 0.5
JAXMD_FIRE_FMAX_HIGH_EVA = 1.0
JAXMD_FIRE_FMAX_VERY_HIGH_EVA = 5.0
# Abort a FIRE stage when live max|F| exceeds best (and stage start) by this factor.
JAXMD_FIRE_BLOWUP_FACTOR = 2.0
# Soft floor so a near-zero best cannot trip blow-up on numerical noise.
JAXMD_FIRE_BLOWUP_ABS_FLOOR_EVA = 0.5
# Also abort when live max|F| rises by this absolute amount vs stage start (eV/Å).
JAXMD_FIRE_BLOWUP_ABS_RISE_EVA = 5.0
# Cap steps/stage on hard starts so a hot inertial stage cannot burn 1000 steps.
JAXMD_FIRE_HARD_START_MAX_STEPS_PER_STAGE = 100
# After FIRE stages still this hard (or blew up without real progress): rebuild
# high-|F| monomers from a healthier peer template, then retry a cold FIRE stage.
JAXMD_FIRE_TEMPLATE_REBUILD_MIN_FMAX_EVA = 2.0
JAXMD_FIRE_TEMPLATE_REBUILD_FORCE_PERCENTILE = 90.0
JAXMD_FIRE_TEMPLATE_REBUILD_MAX_MONOMERS = 64
JAXMD_FIRE_TEMPLATE_REBUILD_RETRY_STEPS = 200
JAXMD_FIRE_BACKOFF_EXTRA_STAGES = 2


def should_skip_jaxmd_fire(
    initial_max_f_eVA: float,
    *,
    skip_below_eVA: float = DEFAULT_JAXMD_FIRE_SKIP_MAX_F_EVA,
) -> bool:
    """True when the start geometry is already soft enough to skip jax-md FIRE."""
    thr = float(skip_below_eVA)
    if thr <= 0.0:
        return False
    f = float(initial_max_f_eVA)
    return bool(np.isfinite(f) and f <= thr)


def should_skip_first_fire_when_pbc_fire_follows(
    *,
    use_pbc: bool,
    first_fire_steps: int,
    pbc_fire_steps: int,
) -> bool:
    """Skip the first FIRE when PBC FIRE will run the same molecular-wrap path.

    Under PBC both stages use monomer wrap + the hybrid force; running the first
    1000-step hot stage then repeating as "PBC FIRE" just wastes wall time
    (seen: max|F| 7→80 for a full stage, then PBC restarts from the early best).
    """
    return bool(
        use_pbc and int(first_fire_steps) > 0 and int(pbc_fire_steps) > 0
    )


def should_skip_redundant_pbc_fire(
    *,
    first_fire_steps: int,
    first_fire_skipped_soft: bool = False,
    first_fire_ran_without_improvement: bool = False,
    use_pbc: bool = True,
) -> bool:
    """Whether PBC FIRE would duplicate work already done by the first FIRE pass.

    Handoff zeros ``jaxmd_minimize_steps`` but keeps ``jaxmd_pbc_minimize_steps``.
    In that case PBC FIRE must still run (``first_fire_steps == 0``).
    """
    if not use_pbc:
        return False
    if int(first_fire_steps) <= 0:
        return False
    return bool(first_fire_skipped_soft or first_fire_ran_without_improvement)


def resolve_jaxmd_fire_stage_steps(
    n_steps: int,
    initial_max_f_eVA: float,
    *,
    hard_cap: int = JAXMD_FIRE_HARD_START_MAX_STEPS_PER_STAGE,
    hard_fmax_eVA: float = JAXMD_FIRE_FMAX_HIGH_EVA,
) -> int:
    """Cap steps/stage when the start is already hard so blow-ups bail quickly."""
    n = max(0, int(n_steps))
    if n <= 0:
        return 0
    f = float(initial_max_f_eVA)
    if np.isfinite(f) and f >= float(hard_fmax_eVA):
        return int(min(n, max(1, int(hard_cap))))
    return n


def resolve_jaxmd_fire_dt_start_ps(initial_max_f_eVA: float) -> float:
    """Choose FIRE ``dt_start`` from how soft the starting geometry already is.

    Hard liquid starts (max|F| ≳ 1 eV/Å) must not inherit the historical 1e-3 ps
    default — that inertial step size drops total energy while a few contacts
    explode (seen: max|F| 7 → 85 on TIP3:903).
    """
    f = float(initial_max_f_eVA)
    if not np.isfinite(f) or f <= 0.0:
        return float(JAXMD_FIRE_DT_SOFT_PS)
    if f < JAXMD_FIRE_FMAX_SOFT_EVA:
        return float(JAXMD_FIRE_DT_SOFT_PS)
    if f < JAXMD_FIRE_FMAX_MILD_EVA:
        return float(JAXMD_FIRE_DT_MILD_PS)
    if f < JAXMD_FIRE_FMAX_HIGH_EVA:
        return float(DEFAULT_JAXMD_FIRE_DT_PS)
    if f < JAXMD_FIRE_FMAX_VERY_HIGH_EVA:
        return float(JAXMD_FIRE_DT_HIGH_F_PS)
    return float(JAXMD_FIRE_DT_VERY_HIGH_F_PS)


def jaxmd_fire_dt_backoff_schedule(
    dt_start_ps: float,
    *,
    n_extra: int = JAXMD_FIRE_BACKOFF_EXTRA_STAGES,
) -> tuple[float, ...]:
    """Descending FIRE dt schedule (restart colder if the first stage wanders)."""
    dt = max(float(dt_start_ps), JAXMD_FIRE_DT_MIN_PS)
    out: list[float] = [dt]
    for _ in range(max(0, int(n_extra))):
        dt = max(dt * 0.3, JAXMD_FIRE_DT_MIN_PS)
        if dt >= out[-1] - 1.0e-15:
            break
        out.append(dt)
    return tuple(out)


def fire_stage_blew_up(
    live_max_f_eVA: float,
    *,
    best_max_f_eVA: float,
    stage_start_max_f_eVA: float,
    blowup_factor: float = JAXMD_FIRE_BLOWUP_FACTOR,
    abs_floor_eVA: float = JAXMD_FIRE_BLOWUP_ABS_FLOOR_EVA,
    abs_rise_eVA: float = JAXMD_FIRE_BLOWUP_ABS_RISE_EVA,
) -> bool:
    """True when live max|F| has exploded vs the stage's best / start."""
    live = float(live_max_f_eVA)
    if not np.isfinite(live):
        return True
    start = float(stage_start_max_f_eVA)
    anchor = max(float(best_max_f_eVA), start, float(abs_floor_eVA))
    if live > float(blowup_factor) * anchor:
        return True
    return bool(live > start + float(abs_rise_eVA))


def should_attempt_fire_template_rebuild(
    fire_info: dict,
    best_max_f: float,
    *,
    min_fmax_eVA: float = JAXMD_FIRE_TEMPLATE_REBUILD_MIN_FMAX_EVA,
) -> bool:
    """Worst-case gate: FIRE still hard after stages, or blew up without real progress."""
    f = float(best_max_f)
    if not np.isfinite(f) or f < float(min_fmax_eVA):
        return False
    stages = list(fire_info.get("stages") or [])
    blew = any(bool(s.get("blew_up")) for s in stages)
    start = float(fire_info.get("start_max_f", f))
    little_progress = f > 0.9 * start
    return bool(blew or little_progress)


def maybe_fire_monomer_template_rebuild_retry(
    *,
    positions,
    best_max_f: float,
    fire_info: dict,
    force_fn,
    energy_fn,
    shift_fn,
    masses,
    monomer_offsets,
    atomic_numbers=None,
    nl_refresh_fn=None,
    log_fn=None,
    console: Console | None = None,
):
    """Rebuild high-|F| monomers from a healthy template, then one cold FIRE retry.

    Donor is geometry-gated (TIP3 HOH/OH); if every peer fails, the bundled
    ideal TIP3 template is used. Returns ``(positions, best_max_f, fire_info)``
    unchanged when the gate is off or no monomers are selected.
    """
    if not should_attempt_fire_template_rebuild(fire_info, best_max_f):
        return positions, best_max_f, fire_info

    pos_np = np.asarray(jax.device_get(positions), dtype=float)
    forces = np.asarray(jax.device_get(force_fn(positions)), dtype=float)
    z_np = None
    if atomic_numbers is not None:
        z_np = np.asarray(jax.device_get(atomic_numbers), dtype=int)
    rebuilt_pos, victims, donor = rebuild_high_force_monomers_from_peers(
        pos_np,
        forces,
        monomer_offsets,
        force_percentile=JAXMD_FIRE_TEMPLATE_REBUILD_FORCE_PERCENTILE,
        max_rebuild=JAXMD_FIRE_TEMPLATE_REBUILD_MAX_MONOMERS,
        min_force_eVA=JAXMD_FIRE_FMAX_HIGH_EVA,
        atomic_numbers=z_np,
    )
    c = console or Console()
    if not victims:
        merged = dict(fire_info)
        merged["template_rebuild"] = {
            "skipped": True,
            "reason": (
                "no force-soft healthy donor "
                f"(gate max|F|<={TEMPLATE_DONOR_MAX_FORCE_EVA:g} eV/Å); "
                "intramolecular template copy will not fix systemic forces"
            ),
            "max_f_before_retry": float(best_max_f),
        }
        c.print(
            Panel(
                f"FIRE still hard (max|F|={best_max_f:.4f} eV/Å): "
                f"skipped template rebuild — no peer with max|F|≤"
                f"{TEMPLATE_DONOR_MAX_FORCE_EVA:g} eV/Å and healthy geometry "
                "(forces look systemic, not crushed monomers).",
                title="[bold yellow]JAX-MD FIRE template rebuild[/bold yellow]",
                border_style="yellow",
            )
        )
        return positions, best_max_f, merged

    if int(donor) == int(TEMPLATE_DONOR_IDEAL_TIP3):
        donor_desc = "ideal TIP3 template (crushed waters; no force-soft peer)"
        donor_source = "ideal_tip3"
    else:
        donor_desc = (
            f"peer template (donor={donor}, "
            f"max|F|≤{TEMPLATE_DONOR_MAX_FORCE_EVA:g} eV/Å gate)"
        )
        donor_source = "peer"

    c.print(
        Panel(
            f"FIRE still hard (max|F|={best_max_f:.4f} eV/Å"
            f"{', blew up' if fire_info.get('blew_up') else ''}): "
            f"rebuilt {len(victims)} monomer(s) from {donor_desc}. "
            f"Cold FIRE retry ({JAXMD_FIRE_TEMPLATE_REBUILD_RETRY_STEPS} steps).",
            title="[bold yellow]JAX-MD FIRE template rebuild[/bold yellow]",
            border_style="yellow",
        )
    )
    retry_pos = jnp.asarray(rebuilt_pos, dtype=getattr(positions, "dtype", jnp.float32))
    if nl_refresh_fn is not None:
        nl_refresh_fn(retry_pos)
    f0 = float(jnp.abs(force_fn(retry_pos)).max())
    dt0 = resolve_jaxmd_fire_dt_start_ps(f0)
    dt_sched = jaxmd_fire_dt_backoff_schedule(dt0)
    retry_pos, retry_f, retry_info = run_jaxmd_fire_with_dt_backoff(
        force_fn=force_fn,
        shift_fn=shift_fn,
        positions=retry_pos,
        masses=masses,
        n_steps=int(JAXMD_FIRE_TEMPLATE_REBUILD_RETRY_STEPS),
        dt_schedule=dt_sched,
        nl_refresh_fn=nl_refresh_fn,
        energy_fn=energy_fn,
        log_fn=log_fn,
    )
    merged = dict(fire_info)
    merged["template_rebuild"] = {
        "n_rebuilt": len(victims),
        "donor": int(donor),
        "donor_source": donor_source,
        "victims": list(victims),
        "max_f_before_retry": float(best_max_f),
        "max_f_after_retry": float(retry_f),
        "retry_stages": retry_info.get("stages"),
    }
    merged["blew_up"] = bool(fire_info.get("blew_up") or retry_info.get("blew_up"))
    merged["best_max_f"] = float(retry_f)
    if float(retry_f) < float(best_max_f):
        return retry_pos, float(retry_f), merged
    # Keep pre-rebuild best if retry did not help forces.
    return positions, best_max_f, merged


def run_jaxmd_fire_with_dt_backoff(
    *,
    force_fn,
    shift_fn,
    positions,
    masses,
    n_steps: int,
    dt_schedule: tuple[float, ...] | list[float],
    worsen_limit: int = WORSE_COUNT_THRESHOLD,
    nl_refresh_fn=None,
    nl_refresh_every: int = JAXMD_FIRE_NL_REFRESH_EVERY,
    log_every: int | None = None,
    log_fn=None,
    energy_fn=None,
    blowup_factor: float = JAXMD_FIRE_BLOWUP_FACTOR,
):
    """FIRE minimize with best-force tracking and smaller-dt restarts.

    Each stage rebuilds ``fire_descent`` at a fixed ``dt_start=dt_max``.  If a
    stage does not improve max|F| vs the geometry it started from, **or** live
    max|F| blows past best/start by ``blowup_factor``, the next colder ``dt`` is
    tried from the best-force frame so far.  An early tiny improvement no longer
    freezes the schedule while mid-stage forces explode.
    """
    if n_steps <= 0:
        pos0 = jnp.asarray(positions)
        f0 = float(jnp.abs(force_fn(pos0)).max())
        return pos0, f0, {
            "stages": [],
            "start_max_f": f0,
            "best_max_f": f0,
            "blew_up": False,
        }

    best_pos = jnp.asarray(positions)
    best_max_f = float(jnp.abs(force_fn(best_pos)).max())
    start_max_f = best_max_f
    stages: list[dict] = []
    print_every = int(log_every) if log_every is not None else max(1, int(n_steps) // 10)
    any_blowup = False

    for stage_idx, dt_ps in enumerate(dt_schedule):
        dt_ps = float(dt_ps)
        stage_start_f = best_max_f
        init_fn, step_fn = jax_md.minimize.fire_descent(
            force_fn,
            shift_fn,
            dt_start=dt_ps,
            dt_max=dt_ps,
        )
        step_fn = jit(step_fn)
        if nl_refresh_fn is not None:
            nl_refresh_fn(best_pos)
        fire_state = init_fn(best_pos, mass=masses)
        worsen_count = 0
        prev_max_f = stage_start_f
        improved = False
        blew_up = False
        steps_run = 0
        for i in range(int(n_steps)):
            steps_run = i + 1
            new_state = step_fn(fire_state)
            if not jnp.all(jnp.isfinite(new_state.position)):
                blew_up = True
                break
            if nl_refresh_fn is not None and nl_refresh_every > 0 and (i + 1) % int(nl_refresh_every) == 0:
                nl_refresh_fn(new_state.position)
            forces = force_fn(new_state.position)
            max_force = float(jnp.abs(forces).max())
            if not np.isfinite(max_force):
                blew_up = True
                break
            if fire_stage_blew_up(
                max_force,
                best_max_f_eVA=best_max_f,
                stage_start_max_f_eVA=stage_start_f,
                blowup_factor=blowup_factor,
            ):
                blew_up = True
                if log_fn is not None:
                    energy = None
                    if energy_fn is not None:
                        try:
                            energy = float(energy_fn(new_state.position))
                        except Exception:
                            energy = None
                    log_fn(stage_idx, dt_ps, i, int(n_steps), energy, max_force)
                break
            fire_state = new_state
            if max_force < best_max_f:
                best_max_f = max_force
                best_pos = fire_state.position
                improved = True
                worsen_count = 0
            else:
                worsen_count = worsen_count + 1 if max_force > prev_max_f else 0
            prev_max_f = max_force
            if log_fn is not None and i % print_every == 0:
                energy = None
                if energy_fn is not None:
                    try:
                        energy = float(energy_fn(fire_state.position))
                    except Exception:
                        energy = None
                log_fn(stage_idx, dt_ps, i, int(n_steps), energy, max_force)
            if worsen_count >= int(worsen_limit):
                break

        any_blowup = any_blowup or blew_up
        stage_improved = bool(improved and best_max_f < stage_start_f - 1.0e-6)
        stages.append(
            {
                "dt_ps": dt_ps,
                "steps_run": steps_run,
                "stage_start_max_f": stage_start_f,
                "best_max_f": best_max_f,
                "improved": stage_improved,
                "blew_up": bool(blew_up),
            }
        )
        # Soft enough: done.  Real progress without blow-up: done.  Otherwise colder dt.
        if best_max_f < 0.05:
            break
        if stage_improved and not blew_up:
            break
        continue

    return best_pos, best_max_f, {
        "stages": stages,
        "start_max_f": start_max_f,
        "best_max_f": best_max_f,
        "dt_final_ps": stages[-1]["dt_ps"] if stages else None,
        "blew_up": bool(any_blowup),
    }


def resolve_mm_pair_list_capacity(
    *,
    update_fn=None,
    pair_idx=None,
) -> int | None:
    """Return MM pair-list padding capacity (number of pair slots).

    ``pair_idx`` is shaped ``(capacity, 2)``; capacity is axis 0.  Using
    ``shape[-1]`` (always 2) produced bogus fill fractions like 74400%.
    Prefer ``update_fn.get_stats()["pair_capacity"]`` when available.
    """
    if update_fn is not None and hasattr(update_fn, "get_stats"):
        try:
            stats = update_fn.get_stats()
            cap = int(stats.get("pair_capacity") or 0)
            if cap > 0:
                return cap
        except Exception:
            pass
    if pair_idx is not None and hasattr(pair_idx, "shape") and len(pair_idx.shape) >= 1:
        # (capacity, 2) → capacity on axis 0; never use shape[-1] (== 2).
        return int(pair_idx.shape[0])
    return None


def resolve_pre_md_fire_start_positions(
    positions,
    masses,
    *,
    use_pbc: bool,
):
    """Starting coordinates for the first JAX-MD FIRE stage.

    Free space: COM-center (historical vacuum FIRE).
    PBC: keep box-frame coordinates — COM-centering plus per-atom periodic
    shift can split monomers across the cell and destroy ASE/CHARMM minima.
    """
    R = jnp.asarray(positions)
    if use_pbc:
        return jnp.asarray(R, dtype=jnp.float32)
    mass = jnp.asarray(masses)
    com = jnp.sum(mass[:, None] * R, axis=0) / mass.sum()
    return jnp.asarray(R - com, dtype=jnp.float32)


# Ensemble-aware PBC MM neighbor refresh when ``--jax-md-update-interval`` is
# omitted / 0. Explicit positive values always win.
#
# NVT can batch more aggressively (thermostat absorbs force noise). NpT needs
# fresher pairs because the cell moves every step. NVE is in between: stale
# pairs show up as E_tot drift, so stay tighter than NVT but not interval=1.
ENSEMBLE_JAXMD_UPDATE_INTERVAL: dict[str, int] = {
    "nvt": 10,
    "npt": 5,
    "nve": 5,
}


def resolve_ensemble_jaxmd_update_interval(
    ensemble: str | None,
    requested: int | None,
    *,
    use_pbc: bool = True,
) -> int:
    """Resolve MM neighbor refresh cadence (MD steps) for a JAX-MD ensemble.

    ``requested <= 0`` or ``None`` selects the ensemble default when ``use_pbc``;
    free-space falls back to a large batch interval (no dynamic MM pairs).
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    if not use_pbc:
        return 100
    key = str(ensemble or "nve").strip().lower()
    return int(ENSEMBLE_JAXMD_UPDATE_INTERVAL.get(key, 1))


def resolve_jaxmd_steps_per_loop_call(
    *,
    steps_per_recording: int,
    use_pbc: bool,
    has_update_fn: bool,
    jax_md_update_interval: int | None,
    ensemble: str | None = None,
) -> int:
    """Return the JAX-MD block size that also controls MM pair refresh cadence.

    The hybrid calculator passes the MM pair list into the compiled integrator
    step as data. In PBC runs with an update function, the interval must
    therefore mean two things at once:

    - how often Python rebuilds the neighbor list;
    - how many MD steps a single compiled JAX block advances before returning.

    When ``jax_md_update_interval`` is ``None``/``0``, the ensemble default from
    :func:`resolve_ensemble_jaxmd_update_interval` is used (NVT batches more than
    NpT/NVE). Explicit positive intervals always win. The final value always
    divides ``steps_per_recording`` so each recording block ends exactly on a
    neighbor-list refresh boundary.
    """
    has_dynamic_pbc_pairs = use_pbc and has_update_fn
    if has_dynamic_pbc_pairs:
        requested_interval = resolve_ensemble_jaxmd_update_interval(
            ensemble, jax_md_update_interval, use_pbc=True
        )
    else:
        requested_interval = (
            int(jax_md_update_interval)
            if jax_md_update_interval is not None and int(jax_md_update_interval) > 0
            else 100
        )

    max_block_steps = min(requested_interval, int(steps_per_recording))
    for candidate_block_steps in range(max_block_steps, 0, -1):
        if int(steps_per_recording) % candidate_block_steps == 0:
            return candidate_block_steps
    return max_block_steps

# Use the same configured precision for the integrator carry and hybrid force
# evaluation.  A float32 carry silently defeated ``--ml-compute-dtype float64``
# and can destabilize Nose-Hoover integration even when the calculator itself
# evaluates in float64.
_JAXMD_DTYPE = resolve_ml_compute_dtype()


def as_jaxmd_dtype(x):
    """Cast to the dtype used for JAX-MD state arrays (positions, forces to integrator)."""
    return jnp.asarray(x, dtype=_JAXMD_DTYPE)


def normalize_jaxmd_state(state):
    """Keep JAX-MD integrator carry dtypes consistent for lax.scan/fori_loop."""
    return state.set(
        position=as_jaxmd_dtype(state.position),
        momentum=as_jaxmd_dtype(state.momentum),
        mass=as_jaxmd_dtype(state.mass),
    )


def _real_cartesian_to_fractional(pos_real: np.ndarray, box_3x3: np.ndarray) -> np.ndarray:
    """Map real-space Cartesian rows (n, 3) to fractional coords for jax_md NPT state."""
    B = np.asarray(box_3x3, dtype=np.float64)[:3, :3]
    R = np.asarray(pos_real, dtype=np.float64)
    invB = np.linalg.inv(B)
    return (invB @ R.T).T

def default_nhc_kwargs(tau, overrides=None):
    """Build Nose-Hoover chain kwargs dict with sensible defaults.

    Args:
        tau: Thermostat coupling timescale (typically ``nhc_tau * dt``).
        overrides: Optional dict to override individual defaults.

    Returns:
        Dict with keys ``chain_length``, ``chain_steps``, ``sy_steps``, ``tau``.
    """
    default_kwargs = {
        'chain_length': 3,
        'chain_steps': 2,
        'sy_steps': 3,
        'tau': tau,
    }
    if overrides is None:
        return default_kwargs
    return {k: overrides.get(k, default_kwargs[k]) for k in default_kwargs}


def _run_npt_diagnostics(
    *,
    state,
    npt_energy_fn,
    jax_md_force_fn,
    apply_fn,
    shift,
    space,
    simulate,
    quantity,
    npt_pair_idx,
    npt_pair_mask,
    npt_pressure,
    unit,
    dt,
    kT,
    grad,
):
    """Run NPT diagnostic tests to locate instabilities. Call with --npt-diagnose."""
    neighbor = (npt_pair_idx, npt_pair_mask)
    box_curr = simulate.npt_box(state)
    R = state.position
    P = state.momentum
    M = state.mass
    N, dim = R.shape

    c = Console()

    # 1. Energy and forces sanity
    E0 = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor))
    real_pos = space.transform(box_curr, R)
    F_calc = jax_md_force_fn(real_pos, mm_pair_idx=npt_pair_idx, mm_pair_mask=npt_pair_mask, box=box_curr)
    F_grad = -grad(lambda r: npt_energy_fn(r, box=box_curr, neighbor=neighbor))(R)
    print_forces_summary(np.asarray(F_calc), energy_eV=E0, console=c)
    t1 = Table(title="[1] Force consistency")
    t1.add_column("Check", style="cyan")
    t1.add_column("Value", style="white")
    t1.add_row("max|F_calc|", f"{float(np.max(np.abs(F_calc))):.6f}")
    t1.add_row("max|F_grad|", f"{float(np.max(np.abs(F_grad))):.6f}")
    t1.add_row("F_calc finite", str(np.all(np.isfinite(F_calc))))
    t1.add_row("F_grad finite", str(np.all(np.isfinite(F_grad))))
    c.print(Panel(t1, title="NPT Diagnostic [1]", border_style="blue"))

    # 2. Perturbation / stress (dUdV)
    vol = float(quantity.volume(dim, box_curr))
    eps_vals = [0.0, 1e-6, 1e-5, 1e-4]
    t2 = Table(title="[2] Stress (dU/dV) via perturbation")
    t2.add_column("ε", style="cyan")
    t2.add_column("E (eV)", style="white")
    for eps in eps_vals:
        pert = 1.0 + eps
        E_pert = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=pert))
        t2.add_row(f"{eps:.0e}", f"{E_pert:.6f}")
    dE = float(npt_energy_fn(R, box=box_curr, neighbor=neighbor, perturbation=1.0 + 1e-5)) - E0
    dUdV_fd = dE / (vol * 1e-5)  # finite-diff approx
    t2.add_row("dUdV (finite diff)", f"{dUdV_fd:.4f} eV/Å³")
    t2.add_row("volume", f"{vol:.2f} Å³")
    c.print(Panel(t2, title="NPT Diagnostic [2]", border_style="blue"))

    # 3. Shift function with fractional R and Cartesian dR
    dR_cart = dt * (P / M)  # small Cartesian displacement
    R_shifted = shift(R, dR_cart, box=box_curr)
    in_cube = np.all((R_shifted >= 0) & (R_shifted < 1.001))
    t3 = Table(title="[3] Shift function")
    t3.add_column("Check", style="cyan")
    t3.add_column("Value", style="white")
    t3.add_row("R_shifted in [0,1)³", str(in_cube))
    t3.add_row("R_shifted finite", str(np.all(np.isfinite(R_shifted))))
    t3.add_row("R_shifted sample [0]", str(np.asarray(R_shifted[0])))
    c.print(Panel(t3, title="NPT Diagnostic [3]", border_style="blue"))

    # 4. exp_iL1-like displacement (barostat scaling term)
    V_b = 0.0  # box velocity at start
    x = V_b * dt
    scale = np.exp(x) - 1
    term1 = R * scale  # fractional * scalar
    term2 = dt * (P / M) * np.exp(x / 2)  # velocity term
    dR_mixed = term1 + term2
    R_after_scale = shift(R, dR_mixed, box=box_curr)
    t4 = Table(title="[4] Barostat scaling term")
    t4.add_column("Check", style="cyan")
    t4.add_column("Value", style="white")
    t4.add_row("x, exp(x)-1", f"{x}, {scale}")
    t4.add_row("max|term1|", f"{float(np.max(np.abs(term1))):.6e}")
    t4.add_row("max|term2|", f"{float(np.max(np.abs(term2))):.6e}")
    t4.add_row("R_after_scale finite", str(np.all(np.isfinite(R_after_scale))))
    t4.add_row("R_after_scale in [0,1)", str(np.all((R_after_scale >= 0) & (R_after_scale < 1.001))))
    c.print(Panel(t4, title="NPT Diagnostic [4]", border_style="blue"))

    # 5. Box and volume
    t5 = Table(title="[5] Box and volume")
    t5.add_column("Property", style="cyan")
    t5.add_column("Value", style="white")
    t5.add_row("box shape", str(np.asarray(box_curr).shape))
    t5.add_row("box diag", str(np.diagonal(np.asarray(box_curr))))
    t5.add_row("box_position (log V/V0)", f"{float(state.box_position)}")
    t5.add_row("box_momentum", f"{float(state.box_momentum)}")
    c.print(Panel(t5, title="NPT Diagnostic [5]", border_style="blue"))

    # 6. State components
    t6 = Table(title="[6] State components")
    t6.add_column("Component", style="cyan")
    t6.add_column("OK", style="white")
    t6.add_row("position finite", str(np.all(np.isfinite(R))))
    t6.add_row("momentum finite", str(np.all(np.isfinite(P))))
    t6.add_row("force finite", str(np.all(np.isfinite(state.force))))
    t6.add_row("mass shape, all positive", f"{M.shape}, {np.all(M > 0)}")
    c.print(Panel(t6, title="NPT Diagnostic [6]", border_style="blue"))

    # 7. Measured vs target pressure (drives box expansion/contraction)
    KE = quantity.kinetic_energy(momentum=P, mass=M)
    t7 = Table(title="[7] Pressure (measured vs target)")
    t7.add_column("Property", style="cyan")
    t7.add_column("Value", style="white")
    try:
        p_meas = quantity.pressure(
            npt_energy_fn, R, box_curr, kinetic_energy=KE,
            neighbor=(npt_pair_idx, npt_pair_mask)
        )
        p_meas_raw = float(p_meas)
        p_tgt_raw = float(npt_pressure)
        BAR_PER_ATM = 1.01325
        unit_p = float(unit["pressure"])
        p_meas_atm = p_meas_raw / (unit_p * BAR_PER_ATM)
        p_tgt_atm = p_tgt_raw / (unit_p * BAR_PER_ATM)
        t7.add_row("P_measured (raw)", f"{p_meas_raw:.6e}")
        t7.add_row("P_target (raw)", f"{p_tgt_raw:.6e}")
        t7.add_row("P_measured (atm)", f"{p_meas_atm:.2f}")
        t7.add_row("P_target (atm)", f"{p_tgt_atm:.2f}")
        t7.add_row("Note", "P_meas > P_tgt → expands; P_meas < P_tgt → contracts")
    except Exception as e:
        t7.add_row("Error", str(e))
    c.print(Panel(t7, title="NPT Diagnostic [7]", border_style="blue"))

    # 8. First step (apply_fn) and NaN location
    neighbor = (npt_pair_idx, npt_pair_mask)
    t8 = Table(title="[8] First NPT step (apply_fn)")
    t8.add_column("Check", style="cyan")
    t8.add_column("Value", style="white")
    try:
        state_one = apply_fn(state, neighbor=neighbor, pressure=npt_pressure)
        pos_ok = np.all(np.isfinite(np.asarray(state_one.position)))
        mom_ok = np.all(np.isfinite(np.asarray(state_one.momentum)))
        box_ok = np.all(np.isfinite(np.asarray(simulate.npt_box(state_one))))
        t8.add_row("position OK", str(pos_ok))
        t8.add_row("momentum OK", str(mom_ok))
        t8.add_row("box OK", str(box_ok))
        if not pos_ok:
            nan_count = np.sum(~np.isfinite(np.asarray(state_one.position)))
            t8.add_row("NaN count", str(nan_count))
            first_nan = np.where(~np.isfinite(np.asarray(state_one.position)))
            if len(first_nan[0]) > 0:
                t8.add_row("First NaN index", f"({first_nan[0][0]}, {first_nan[1][0]})")
    except Exception as e:
        t8.add_row("Error", f"{type(e).__name__}: {e}")
    c.print(Panel(t8, title="NPT Diagnostic [8]", border_style="blue"))
    c.print(Panel("NPT diagnostic complete", title="[bold]NPT Diagnostics (--npt-diagnose)[/bold]", border_style="green"))


def set_up_nhc_sim_routine(
    atoms,
    args,
    spherical_cutoff_calculator,
    get_update_fn,
    CUTOFF_PARAMS,
    n_monomers,
    monomer_offsets,
    Si_mass,
    show_frame=None,
    atoms_template=None,
    overlap_charmm_rescue_fn: Optional[
        Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
    ] = None,
    initial_velocities: Optional[np.ndarray] = None,
    minimization_skipped: bool = False,
):
    """Set up the Nose-Hoover chain simulation routine.

    Returns:
        The run_sim function.
    """
    atoms_template = atoms_template if atoms_template is not None else atoms
    T = args.temperature
    Si_mass = as_jaxmd_dtype(Si_mass)

    @jax.jit
    def evaluate_energies_and_forces(
        atomic_numbers,
        positions,
        mm_pair_idx=None,
        mm_pair_mask=None,
        box=None,
    ):
        return spherical_cutoff_calculator(
            atomic_numbers=atomic_numbers,
            positions=positions,
            n_monomers=n_monomers,
            cutoff_params=CUTOFF_PARAMS,
            doML=True,
            doMM=args.include_mm,
            doML_dimer=not args.skip_ml_dimers,
            debug=args.debug,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )

    atomic_numbers = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
    R = as_jaxmd_dtype(atoms.get_positions())

    @jit
    def jax_md_eval_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
        position = as_jaxmd_dtype(position)
        return evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=position,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )

    @jit
    def jax_md_energy_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
        return jax_md_eval_fn(
            position,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
            **kwargs,
        ).energy.reshape(-1)[0]

    @jit
    def jax_md_force_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
        """Return forces from calculator (no autodiff). jax.grad(energy_fn) produces NaN."""
        position = as_jaxmd_dtype(position)
        result = evaluate_energies_and_forces(
            atomic_numbers=atomic_numbers,
            positions=position,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
            box=box,
        )
        return as_jaxmd_dtype(result.forces)

    # Optional PSF/CGenFF angle (+ Urey) restraints — keep ML monomers tetrahedral
    # when hybrid ml_intra has no classical bonded MM / SHAKE.
    _psf_angle_on = bool(getattr(args, "psf_angle_restraints", False))
    _psf_angle_energy_fn = None
    _psf_angle_force_fn = None
    if _psf_angle_on:
        from mmml.md.restraints.psf_angles import build_psf_angle_restraint_fns

        _psf_path = getattr(args, "from_psf", None)
        if _psf_path is None:
            raise ValueError("--psf-angle-restraints requires --from-psf")
        _box_A = float(args.cell) if getattr(args, "cell", None) else None
        _psf_angle_energy_fn, _psf_angle_force_fn, _psf_angle_info = (
            build_psf_angle_restraint_fns(
                _psf_path,
                R,
                box_A=_box_A,
                scale=float(getattr(args, "psf_angle_restraint_scale", 1.0) or 1.0),
                include_urey=not bool(
                    getattr(args, "psf_angle_restraints_no_urey", False)
                ),
            )
        )
        _base_energy_fn = jax_md_energy_fn
        _base_force_fn = jax_md_force_fn

        @jit
        def jax_md_energy_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
            e0 = _base_energy_fn(
                position,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
                **kwargs,
            )
            return e0 + _psf_angle_energy_fn(as_jaxmd_dtype(position))

        @jit
        def jax_md_force_fn(position, mm_pair_idx=None, mm_pair_mask=None, box=None, **kwargs):
            f0 = _base_force_fn(
                position,
                mm_pair_idx=mm_pair_idx,
                mm_pair_mask=mm_pair_mask,
                box=box,
                **kwargs,
            )
            return f0 + as_jaxmd_dtype(_psf_angle_force_fn(as_jaxmd_dtype(position)))

        c0 = Console()
        c0.print(
            Panel(
                f"PSF={_psf_angle_info.psf_path}\n"
                f"angles={_psf_angle_info.n_angles}  urey={_psf_angle_info.n_urey}  "
                f"scale={_psf_angle_info.scale:g}  box={_psf_angle_info.box_A}",
                title="[bold]PSF angle restraints (tetrahedral)[/bold]",
                border_style="magenta",
            )
        )

    # evaluate_energies_and_forces (initial call - get update_fn if available)
    use_pbc = args.cell is not None
    is_npt = args.ensemble == "npt" and use_pbc
    update_fn = get_update_fn(R, CUTOFF_PARAMS) if get_update_fn else None
    pair_idx, pair_mask = None, None
    # Use (3,) or 3x3 box format for consistency with mm_energy_forces._box_to_cell_3x3
    L_cell = float(args.cell) if args.cell else None
    box_init = jnp.array([L_cell, L_cell, L_cell], dtype=_JAXMD_DTYPE) if L_cell else None
    box_nl = np.array([L_cell, L_cell, L_cell], dtype=np.float64) if L_cell else None
    pbc_box_nl = box_nl  # Capture for run_sim PBC minimization (avoids UnboundLocalError from later box_nl assignments)
    if update_fn is not None and use_pbc:
        if getattr(args, "debug", False):
            print("[nbr] Initial neighbor list update (PBC)")
        if is_npt:
            # NPT: neighbor list uses fractional_coordinates; pass frac pos and box [L,L,L]
            R_frac = np.asarray(R) / L_cell
            pair_idx, pair_mask = update_fn(R_frac, box=box_nl)
        else:
            # NVT/NVE: fixed box, pass box for neighbor list consistency
            pair_idx, pair_mask = update_fn(R, box=box_nl)
    c = Console()
    # Silent compile + GPU sync before timed run (avoids XLA cuda_timer delay-kernel warnings).
    ensure_xla_gpu_warmed(force=True)
    _warm = evaluate_energies_and_forces(
        atomic_numbers=atomic_numbers,
        positions=R,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    block_jax_values(_warm.energy, _warm.forces)
    c.print(Panel("Compiling JAX energy/force (first run may take minutes)...", title="[bold cyan]JAX-MD[/bold cyan]", border_style="cyan"))
    t0 = time.perf_counter()
    result = evaluate_energies_and_forces(
        atomic_numbers=atomic_numbers,
        positions=R,
        mm_pair_idx=pair_idx,
        mm_pair_mask=pair_mask,
        box=box_init,
    )
    elapsed = time.perf_counter() - t0
    init_energy = result.energy.reshape(-1)[0]
    init_forces = np.asarray(result.forces).reshape(-1, 3)
    flat_bottom_radius = getattr(args, "flat_bottom_radius", None)
    flat_bottom_k = float(getattr(args, "flat_bottom_k", 1.0))
    flat_bottom_mode = str(getattr(args, "flat_bottom_mode", "system")).lower().strip()
    report_wall = True  # the short-range wall is on by default in the calculator
    use_flat_bottom = (
        flat_bottom_radius is not None and float(flat_bottom_radius) > 0.0
    )
    _fb_dist_hdr = (
        "max|COM_m| (Å)" if flat_bottom_mode == "monomer" else "|COM| (Å)"
    )
    c.print(Panel(f"Compilation done in {elapsed:.2f} s", title="[bold green]JAX[/bold green]", border_style="green"))

    # setup_calculator already emitted Track A+B for md-system. After compile we
    # only refresh the neighbor-list panel with live capacity / fill fractions.
    _checkpoint_hint = str(getattr(args, "checkpoint", None) or getattr(args, "output_prefix", ""))
    _nl_n_valid = None
    _nl_skin = getattr(args, "jax_md_skin_distance", None)
    _nl_interval = getattr(args, "jax_md_update_interval", None)
    _nl_capacity = resolve_mm_pair_list_capacity(update_fn=update_fn, pair_idx=pair_idx)
    if pair_mask is not None:
        try:
            _nl_n_valid = int(np.sum(np.asarray(pair_mask)))
        except Exception:
            _nl_n_valid = None
    if use_pbc and (pair_idx is not None or _nl_capacity is not None):
        emit_md_system_calculator_report(
            cutoff_params=CUTOFF_PARAMS,
            n_monomers=n_monomers,
            n_atoms=len(atoms),
            cell_L_A=float(args.cell) if args.cell is not None else None,
            mm_cutoff_A=float(CUTOFF_PARAMS.mm_switch_on + CUTOFF_PARAMS.mm_switch_width),
            capacity_pairs=_nl_capacity,
            n_valid_pairs=_nl_n_valid,
            skin_distance_A=float(_nl_skin) if _nl_skin is not None else None,
            update_interval_steps=int(_nl_interval) if _nl_interval is not None else None,
            include_hybrid_setup=False,
            include_calculator_summary=False,
            include_neighbor_list_summary=True,
            include_psf_topology=False,
        )

    # Save calculator summary JSON to run directory
    _run_prefix = Path(str(getattr(args, "output_prefix", "md")))
    _calc_json_path = _run_prefix.parent / "calculator_summary.json"
    try:
        save_calculator_summary_json(
            _calc_json_path,
            CUTOFF_PARAMS,
            model_type="Hybrid ML/MM (spherical_cutoff_calculator)",
            n_monomers=n_monomers,
            n_atoms=len(atoms),
            doML=True,
            doMM=getattr(args, "include_mm", True),
            doML_dimer=not getattr(args, "skip_ml_dimers", False),
            ensemble=getattr(args, "ensemble", None),
            checkpoint=_checkpoint_hint,
            nl_capacity_pairs=_nl_capacity,
            nl_n_valid_pairs=_nl_n_valid,
            nl_skin_distance_A=float(_nl_skin) if _nl_skin is not None else None,
            nl_update_interval_steps=int(_nl_interval) if _nl_interval is not None else None,
        )
        c.print(Panel(str(_calc_json_path), title="[bold green]Calculator Summary JSON[/bold green]", border_style="green"))
    except Exception as _e:
        c.print(Panel(f"Could not save calculator_summary.json: {_e}", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))

    _eval_label = (
        "post-compile (initial R)"
        if minimization_skipped
        else "post-compile (ASE-minimized R)"
    )
    print_forces_summary(init_forces, energy_eV=float(init_energy), console=c)
    try:
        from mmml.analysis.hybrid_force_breakdown import (
            hybrid_force_term_breakdown,
            print_hybrid_force_term_breakdown,
            write_hybrid_force_term_breakdown_json,
        )

        _fbreak = hybrid_force_term_breakdown(
            result,
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        )
        print_hybrid_force_term_breakdown(
            _fbreak,
            title=f"Hybrid force-term breakdown ({_eval_label})",
        )
        _fbreak_path = _run_prefix.parent / "force_term_breakdown.json"
        write_hybrid_force_term_breakdown_json(_fbreak, _fbreak_path)
        c.print(
            Panel(
                str(_fbreak_path),
                title="[bold green]Force-term breakdown JSON[/bold green]",
                border_style="green",
            )
        )
    except Exception as _fbreak_err:
        c.print(
            Panel(
                f"Could not build force-term breakdown: {_fbreak_err}",
                title="[bold yellow]Warning[/bold yellow]",
                border_style="yellow",
            )
        )
    print_flat_bottom_summary(
        result,
        flat_bottom_radius=flat_bottom_radius,
        flat_bottom_k=flat_bottom_k,
        flat_bottom_mode=flat_bottom_mode,
        label=_eval_label,
        console=c,
    )

    # MIC-only PBC: calculator uses minimum-image convention, no coordinate transform.
    pbc_map_fn = getattr(atoms.calc, "pbc_map", None) if atoms.calc else None
    pbc_info = f"BOXSIZE: {float(args.cell)} Å, PBC: True (MIC-only)" if use_pbc else "free space (no PBC), pbc_map: False"
    c.print(Panel(pbc_info, title="[bold]JAX-MD PBC[/bold]", border_style="blue"))

    # Mutable container for box/pairs so PBC minimization can update pairs for pbc_start_pos
    _pbc_state = {"box": box_init, "pair_idx": pair_idx, "pair_mask": pair_mask}

    def _eval_at_position(position, *, box=None, pair_idx=None, pair_mask=None):
        return jax_md_eval_fn(
            position,
            mm_pair_idx=pair_idx if pair_idx is not None else _pbc_state["pair_idx"],
            mm_pair_mask=pair_mask if pair_mask is not None else _pbc_state["pair_mask"],
            box=box if box is not None else _pbc_state["box"],
        )

    # Energy and force: use calculator's explicit forces (jax.grad through calculator gives NaN).
    # MIC-only PBC: no coordinate transform; calculator uses MIC internally.
    if use_pbc and pbc_map_fn is not None:
        @jax.custom_vjp
        def wrapped_energy_fn(position, **kwargs):
            pos = jnp.array(position)
            neighbor = kwargs.get("neighbor", None)
            pair_idx, pair_mask = neighbor if neighbor is not None else (_pbc_state["pair_idx"], _pbc_state["pair_mask"])
            return jax_md_energy_fn(
                pbc_map_fn(pos),
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )

        def wrapped_energy_fn_fwd(position, **kwargs):
            pos = jnp.array(position)
            R_mapped = pbc_map_fn(pos)
            neighbor = kwargs.get("neighbor", None)
            pair_idx, pair_mask = neighbor if neighbor is not None else (_pbc_state["pair_idx"], _pbc_state["pair_mask"])
            E = jax_md_energy_fn(
                R_mapped,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )
            return E, (pos, R_mapped, pair_idx, pair_mask)

        def wrapped_energy_fn_bwd(res, g, **kwargs):
            pos, R_mapped, pair_idx, pair_mask = res
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=R_mapped,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )
            F_mapped = result.forces
            F_orig = pbc_map_fn.transform_forces(pos, F_mapped)
            return (F_orig,)

        wrapped_energy_fn.defvjp(wrapped_energy_fn_fwd, wrapped_energy_fn_bwd)
        wrapped_energy_fn = jit(wrapped_energy_fn)

        @jit
        def wrapped_force_fn(position, **kwargs):
            pos = jnp.array(position)
            R_mapped = pbc_map_fn(pos)
            neighbor = kwargs.get("neighbor", None)
            pair_idx, pair_mask = neighbor if neighbor is not None else (_pbc_state["pair_idx"], _pbc_state["pair_mask"])
            F_mapped = jax_md_force_fn(
                R_mapped,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )
            return as_jaxmd_dtype(pbc_map_fn.transform_forces(pos, F_mapped))
    else:
        # MIC-only: capture box and pairs for PBC minimization (Fix A)
        @jit
        def wrapped_energy_fn(position, **kwargs):
            neighbor = kwargs.get("neighbor", None)
            pair_idx, pair_mask = neighbor if neighbor is not None else (_pbc_state["pair_idx"], _pbc_state["pair_mask"])
            return jax_md_energy_fn(
                jnp.array(position),
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )

        @jit
        def wrapped_force_fn(position, **kwargs):
            neighbor = kwargs.get("neighbor", None)
            pair_idx, pair_mask = neighbor if neighbor is not None else (_pbc_state["pair_idx"], _pbc_state["pair_mask"])
            return jax_md_force_fn(
                jnp.array(position),
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )

    # Shift and displacement for minimization and simulation
    # Minimization: energy/force use Cartesian (MIC). Use space.periodic when PBC so positions
    # stay in box; avoids coordinate mismatch (fractional shift + Cartesian energy → oscillation).
    # Simulation: NPT uses fractional; NVT/NVE use free or periodic.
    is_npt = args.ensemble == "npt" and use_pbc
    L_cell_val = float(args.cell) if args.cell else None
    if is_npt:
        L_npt = L_cell_val
        box_npt = jnp.eye(3, dtype=jnp.float32) * L_npt
        displacement, shift = space.periodic_general(box=box_npt, fractional_coordinates=True)
    else:
        displacement, shift = space.free()

    # Free-space FIRE uses free shift.  Under PBC the first FIRE is built inside
    # run_sim with molecular (monomer-COM) wrapping — never space.periodic, which
    # wraps atoms individually and can split monomers across the box.
    shift_min = shift

    # ========================================================================
    # SIMULATION PARAMETERS (metal units: eV, Å, ps, amu)
    # ========================================================================
    unit = units.metal_unit_system()
    # dt must be in ps: args.timestep is fs, 1 fs = 0.001 ps
    dt_fs = args.timestep
    dt = dt_fs * 0.001
    # NPT: neighbor list must be updated frequently (box changes every step).
    # Using 1000 steps with a stale neighbor list causes wrong forces → NaN.
    steps_per_recording = (
        getattr(args, "steps_per_recording", None)
        or (25 if (args.ensemble == "npt" and use_pbc) else 1000)
    )
    steps_per_loop_call = resolve_jaxmd_steps_per_loop_call(
        steps_per_recording=int(steps_per_recording),
        use_pbc=bool(use_pbc),
        has_update_fn=get_update_fn is not None,
        jax_md_update_interval=getattr(args, "jax_md_update_interval", None),
        ensemble=getattr(args, "ensemble", None),
    )

    kT = as_jaxmd_dtype(T * unit['temperature'])
    jax.random.PRNGKey(0)
    c.print(Panel(
        f"Ensemble: {args.ensemble.upper()} | dt={dt} ps ({dt_fs} fs) | kT={kT} ({T} K) | steps_per_recording={steps_per_recording} | steps_per_loop_call={steps_per_loop_call}",
        title="[bold]JAX-MD Simulation[/bold]",
        border_style="cyan",
    ))

    def _bind_sim(apply_fn_local):
        """(Re)build the jitted multi-step integrator bound to ``apply_fn_local``."""

        @jit
        def _sim(state, neighbor=None, pressure=None):
            def _cast_state(s):
                return normalize_jaxmd_state(s)

            def step_nve(i, s):
                if neighbor is not None:
                    return _cast_state(apply_fn_local(s, neighbor=neighbor))
                return _cast_state(apply_fn_local(s))

            def step_npt(i, s):
                return _cast_state(
                    apply_fn_local(s, neighbor=neighbor, pressure=pressure)
                )

            step_fn = (
                step_npt
                if (neighbor is not None and pressure is not None)
                else step_nve
            )
            return lax.fori_loop(0, steps_per_loop_call, step_fn, state)

        return _sim

    # Select integrator based on ensemble
    if args.ensemble == "npt" and use_pbc:
        if update_fn is None:
            raise ValueError(
                "NPT requires jax_md neighbor list (cell list cannot handle dynamic box). "
                "Ensure jax_md is installed and pbc_cell is set."
            )
        BAR_PER_ATM = 1.01325
        p_atm = getattr(args, 'pressure', 1.0)
        if p_atm <= 0:
            # Preserve initial density: P = N*kT/V (ideal gas) so box stays ~constant
            V_init = float(L_cell_val ** 3)
            p_atm = float(n_monomers * kT / V_init / (unit['pressure'] * BAR_PER_ATM))
            c.print(Panel(f"pressure=0 → density-preserving P={p_atm:.2f} atm (N={n_monomers}, V={V_init:.0f} Å³)", title="[bold]NPT[/bold]", border_style="yellow"))
        # Pressure for npt_nose_hoover: jax_md uses same units as energy/volume.
        # Metal: energy=eV, V=Å³ → pressure in eV/Å³. 1 bar = unit['pressure'] eV/Å³; 1 atm = 1.01325 bar.
        pressure = jnp.array(p_atm * BAR_PER_ATM * unit['pressure'], dtype=jnp.float32)
        # Barostat tau: 10000*dt (2.5 ps at 0.25 fs) avoids NaN from aggressive box scaling
        barostat_tau = getattr(args, 'nhc_barostat_tau', 10000.0) * dt
        nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
        nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
        nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
        nhc_tau = getattr(args, 'nhc_tau', 100.0)
        nhc_kwargs = {
            'chain_length': nhc_chain_length,
            'chain_steps': nhc_chain_steps,
            'sy_steps': nhc_sy_steps,
        }

        def _npt_energy_fn_raw(frac_pos, box=None, neighbor=None, perturbation=None, **kwargs):
            """Energy in fractional coords: transform to real, then evaluate.
            Supports perturbation=(1+eps) for NPT barostat stress (dU/dV)."""
            box_eff = jnp.asarray(box, dtype=jnp.float32)
            if perturbation is not None:
                # Isotropic: V' = V * perturbation, so L' = L * perturbation^(1/3)
                scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                box_eff = box_eff * scale
            real_pos = space.transform(box_eff, frac_pos)
            pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
            result = evaluate_energies_and_forces(
                atomic_numbers=atomic_numbers,
                positions=real_pos,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=box_eff,
            )
            return result.energy.reshape(-1)[0]

        @jax.custom_vjp
        def npt_energy_fn(frac_pos, box=None, neighbor=None, perturbation=None, kT=None, mass=None):
            """NPT energy with custom VJP: use explicit calculator forces (jax.grad gives NaN).
            All kwargs as explicit params so JAX resolve_kwargs can bind them to positions."""
            return _npt_energy_fn_raw(
                frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
            )

        def npt_energy_fn_fwd(frac_pos, box, neighbor, perturbation, kT, mass):
            E = _npt_energy_fn_raw(
                frac_pos, box=box, neighbor=neighbor, perturbation=perturbation
            )
            return E, (frac_pos, box, neighbor, perturbation)

        def npt_energy_fn_bwd(res, g):
            frac_pos, box, neighbor, perturbation = res
            box_eff = jnp.asarray(box, dtype=jnp.float32)
            if perturbation is not None:
                scale = jnp.power(jnp.asarray(perturbation, dtype=jnp.float32), 1.0 / 3.0)
                box_eff = box_eff * scale
            real_pos = space.transform(box_eff, frac_pos)
            pair_idx, pair_mask = neighbor if neighbor is not None else (None, None)
            F = jax_md_force_fn(
                real_pos,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=box_eff,
            )
            # grad(E) = -F; quantity.force = -grad, so we supply -F as grad
            grad_frac = as_jaxmd_dtype(-F * g)
            return (grad_frac, None, None, None, None, None)

        npt_energy_fn.defvjp(npt_energy_fn_fwd, npt_energy_fn_bwd)
        npt_energy_fn = jit(npt_energy_fn)
        init_fn, apply_fn = simulate.npt_nose_hoover(
            npt_energy_fn,
            shift,
            dt=dt,
            pressure=pressure,
            kT=kT,
            barostat_kwargs=default_nhc_kwargs(as_jaxmd_dtype(barostat_tau), nhc_kwargs),
            thermostat_kwargs=default_nhc_kwargs(as_jaxmd_dtype(nhc_tau * dt), nhc_kwargs),
        )
        c.print(Panel(
            f"pressure={p_atm:.2f} atm | barostat_tau={barostat_tau:.6f} ps | thermostat tau={nhc_tau * dt:.6f} ps",
            title="[bold]NPT Nose-Hoover[/bold]",
            border_style="green",
        ))
    elif args.ensemble == "nvt":
        nhc_chain_length = getattr(args, 'nhc_chain_length', 3)
        nhc_chain_steps = getattr(args, 'nhc_chain_steps', 2)
        nhc_sy_steps = getattr(args, 'nhc_sy_steps', 3)
        nhc_tau = getattr(args, 'nhc_tau', 100.0)
        nhc_kwargs = {
            'chain_length': nhc_chain_length,
            'chain_steps': nhc_chain_steps,
            'sy_steps': nhc_sy_steps,
        }
        init_fn, apply_fn = simulate.nvt_nose_hoover(
            wrapped_force_fn, shift, dt=dt, kT=kT,
            thermostat_kwargs=default_nhc_kwargs(
                as_jaxmd_dtype(nhc_tau * dt), nhc_kwargs
            ),
        )
        c.print(Panel(
            f"chain_length={nhc_chain_length} | chain_steps={nhc_chain_steps} | sy_steps={nhc_sy_steps} | tau={nhc_tau * dt:.6f} ps",
            title="[bold]NVT Nose-Hoover[/bold]",
            border_style="green",
        ))
    else:  # nve
        init_fn, apply_fn = simulate.nve(wrapped_force_fn, shift, dt)
    apply_fn = jit(apply_fn)
    sim = _bind_sim(apply_fn)

    def run_sim(
        key,
        total_steps=args.nsteps_jaxmd,
        steps_per_recording=steps_per_recording,
        R=R,
        skip_minimization=False,
    ):
        # May be rebound mid-NVE on E_tot drift dt_halve rescue.
        nonlocal init_fn, apply_fn, sim, dt, dt_fs, steps_per_loop_call
        run_sim.last_status = "running"
        run_sim.last_error = None
        run_sim.last_hdf5_path = None
        run_sim.last_velocities = None
        total_records = total_steps // steps_per_recording
        _monomer_groups = [
            jnp.arange(int(monomer_offsets[m]), int(monomer_offsets[m + 1]))
            for m in range(n_monomers)
        ]
        _monomer_group_id = group_ids_from_groups(_monomer_groups, n_atoms=len(atoms))
        _monomer_mass_sum = jax.ops.segment_sum(
            Si_mass,
            _monomer_group_id,
            num_segments=n_monomers,
        )

        @jit
        def _wrap_monomers(positions, cell):
            return wrap_groups_by_id_with_weight_sum(
                positions,
                _monomer_group_id,
                _monomer_mass_sum,
                cell,
                mass=Si_mass,
            )

        overlap_min_distance = float(getattr(args, "min_intermonomer_atom_distance", 0.1))
        overlap_action = str(getattr(args, "dynamics_overlap_action", "warn")).lower()
        # Optional path: molecular wrapping every frame at HDF5/export time (better viz).
        traj_export_molecular_wrap = bool(getattr(args, "traj_export_molecular_wrap", False))
        overlap_warning_count = 0
        overlap_min_seen = float("inf")
        charmm_overlap_rescue_count = 0

        def _check_overlap(
            positions, cell, context: str
        ) -> Optional[np.ndarray]:
            """Return new real-space Cartesian positions if CHARMM rescue was applied."""
            nonlocal overlap_warning_count, overlap_min_seen, charmm_overlap_rescue_count
            if overlap_action == "off":
                return None
            try:
                min_dist = assert_no_intermonomer_atom_overlap(
                    np.asarray(jax.device_get(positions), dtype=float),
                    monomer_offsets,
                    min_distance=overlap_min_distance,
                    cell=None if cell is None else np.asarray(jax.device_get(cell), dtype=float),
                    context=context,
                )
                overlap_min_seen = min(overlap_min_seen, min_dist)
                return None
            except RuntimeError as exc:
                overlap_warning_count += 1
                message = str(exc)
                try:
                    min_dist = float(message.split("distance=")[1].split(" A")[0])
                    overlap_min_seen = min(overlap_min_seen, min_dist)
                except (IndexError, ValueError):
                    pass
                if overlap_action == "error":
                    raise
                if (
                    overlap_action in ("warn", "rescue")
                    and overlap_charmm_rescue_fn is not None
                ):
                    pos_np = np.asarray(jax.device_get(positions), dtype=float)
                    cell_np = (
                        None
                        if cell is None
                        else np.asarray(jax.device_get(cell), dtype=float)
                    )
                    try:
                        new_pos = overlap_charmm_rescue_fn(pos_np, cell_np)
                        charmm_overlap_rescue_count += 1
                        c.print(Panel(
                            f"{message}\nApplied CHARMM SD/ABNR overlap rescue (box synced to MD cell); "
                            "re-initializing Maxwell–Boltzmann velocities at target T.",
                            title="[bold green]JAX-MD overlap → CHARMM rescue[/bold green]",
                            border_style="green",
                        ))
                        return np.asarray(new_pos, dtype=float)
                    except Exception as rescue_exc:
                        c.print(Panel(
                            f"{message}\nCHARMM rescue failed ({type(rescue_exc).__name__}: {rescue_exc}).",
                            title="[bold red]JAX-MD overlap rescue failed[/bold red]",
                            border_style="red",
                        ))
                if overlap_warning_count <= 5 or overlap_warning_count % 50 == 0:
                    c.print(Panel(
                        f"{message}\nContinuing because dynamics_overlap_action={overlap_action!r}.",
                        title="[bold yellow]JAX-MD overlap warning[/bold yellow]",
                        border_style="yellow",
                    ))
                return None
        fire_positions = []
        skip_redundant_pbc_fire = False
        fire_skip_thr = float(
            getattr(args, "jaxmd_fire_skip_max_f_eVA", DEFAULT_JAXMD_FIRE_SKIP_MAX_F_EVA)
        )
        if skip_minimization:
            minimized_pos = jnp.asarray(R, dtype=jnp.float32)
            nmin_pbc_planned = int(getattr(args, "jaxmd_pbc_minimize_steps", 0) or 0)
            if use_pbc and nmin_pbc_planned > 0:
                skip_msg = (
                    "Skipping first FIRE (handoff positions); "
                    f"PBC FIRE ({nmin_pbc_planned} steps) follows."
                )
            else:
                skip_msg = "Skipping minimization (using input positions)"
            c.print(Panel(skip_msg, title="[bold]JAX-MD Minimization[/bold]", border_style="yellow"))
        else:
            initial_pos = resolve_pre_md_fire_start_positions(
                R, Si_mass, use_pbc=bool(use_pbc)
            )
            fire_shift_fn = shift_min
            if use_pbc:
                # Molecular wrap (same policy as PBC FIRE below). Never use
                # space.periodic here — per-atom wrap splits monomers.
                _cell_fire = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
                initial_pos = _wrap_monomers(initial_pos, _cell_fire)
                if update_fn is not None:
                    fire_pair_idx, fire_pair_mask = update_fn(
                        np.asarray(initial_pos), box=pbc_box_nl
                    )
                    _pbc_state["pair_idx"] = fire_pair_idx
                    _pbc_state["pair_mask"] = fire_pair_mask

                def fire_shift_fn(pos, dR, **kwargs):
                    return _wrap_monomers(pos + dR, _cell_fire)

            # Sanity check: ensure energy/gradient are finite at start
            try:
                _out0 = jax_md_eval_fn(
                    initial_pos,
                    mm_pair_idx=_pbc_state["pair_idx"],
                    mm_pair_mask=_pbc_state["pair_mask"],
                    box=_pbc_state["box"],
                )
                _e0 = float(_out0.energy)
                _f0 = _out0.forces
                if not (np.isfinite(_e0) and np.all(np.isfinite(np.asarray(_f0)))):
                    initial_pos = jnp.asarray(R, dtype=jnp.float32)
                    if use_pbc:
                        initial_pos = _wrap_monomers(
                            initial_pos, jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
                        )
                    c.print(Panel(
                        "Non-finite energy/forces at FIRE start; using wrapped/raw R",
                        title="[bold yellow]Warning[/bold yellow]",
                        border_style="yellow",
                    ))
                else:
                    print_flat_bottom_summary(
                        _out0,
                        flat_bottom_radius=flat_bottom_radius,
                        flat_bottom_k=flat_bottom_k,
                        flat_bottom_mode=flat_bottom_mode,
                        label=(
                            "FIRE start (molecular wrap, box frame)"
                            if use_pbc
                            else "FIRE start (COM-centered)"
                        ),
                        console=c,
                    )
                if not use_pbc:
                    _out_r = jax_md_eval_fn(
                        R,
                        mm_pair_idx=_pbc_state["pair_idx"],
                        mm_pair_mask=_pbc_state["pair_mask"],
                        box=_pbc_state["box"],
                    )
                    print_flat_bottom_summary(
                        _out_r,
                        flat_bottom_radius=flat_bottom_radius,
                        flat_bottom_k=flat_bottom_k,
                        flat_bottom_mode=flat_bottom_mode,
                        label="FIRE reference (raw R, no COM shift)",
                        console=c,
                    )
            except Exception:
                initial_pos = jnp.asarray(R, dtype=jnp.float32)
                print("Fallback: using R directly for minimization")

            NMIN = getattr(args, "jaxmd_minimize_steps", 1000)
            NMIN_PBC_PLANNED = int(getattr(args, "jaxmd_pbc_minimize_steps", 0) or 0)
            skip_first_for_pbc = should_skip_first_fire_when_pbc_fire_follows(
                use_pbc=bool(use_pbc),
                first_fire_steps=int(NMIN),
                pbc_fire_steps=NMIN_PBC_PLANNED,
            )
            if NMIN <= 0 or skip_first_for_pbc:
                why = (
                    "PBC FIRE follows with the same molecular-wrap path"
                    if skip_first_for_pbc
                    else "0 steps requested"
                )
                c.print(
                    Panel(
                        f"Skipping first minimization ({why}).",
                        title="[bold]JAX-MD Minimization[/bold]",
                        border_style="yellow",
                    )
                )
                minimized_pos = initial_pos
                skip_redundant_pbc_fire = should_skip_redundant_pbc_fire(
                    first_fire_steps=0, use_pbc=bool(use_pbc)
                )
            else:
                f0_comp = float(jnp.abs(wrapped_force_fn(initial_pos)).max())
                if should_skip_jaxmd_fire(f0_comp, skip_below_eVA=fire_skip_thr):
                    minimized_pos = initial_pos
                    skip_redundant_pbc_fire = should_skip_redundant_pbc_fire(
                        first_fire_steps=int(NMIN),
                        first_fire_skipped_soft=True,
                        use_pbc=bool(use_pbc),
                    )
                    c.print(
                        Panel(
                            f"Skipping jax-md FIRE: start max|F|={f0_comp:.4f} eV/Å "
                            f"≤ skip gate {fire_skip_thr:.4f} eV/Å "
                            "(ASE/CHARMM already soft). "
                            "Use --jaxmd-fire-skip-max-f-eVA 0 to force FIRE.",
                            title="[bold]JAX-MD Minimization[/bold]",
                            border_style="yellow",
                        )
                    )
                else:
                    n_fire = resolve_jaxmd_fire_stage_steps(int(NMIN), f0_comp)
                    dt0 = resolve_jaxmd_fire_dt_start_ps(f0_comp)
                    dt_sched = jaxmd_fire_dt_backoff_schedule(dt0)
                    fire_label = (
                        f"FIRE minimization ({n_fire} steps/stage, molecular wrap; "
                        f"dt schedule={[f'{d:.1e}' for d in dt_sched]} ps)"
                        if use_pbc
                        else (
                            f"FIRE minimization ({n_fire} steps/stage; "
                            f"dt schedule={[f'{d:.1e}' for d in dt_sched]} ps)"
                        )
                    )
                    c.print(Panel(fire_label, title="[bold cyan]JAX-MD Minimization[/bold cyan]", border_style="cyan"))

                    def _fire_nl_refresh(pos):
                        if use_pbc and update_fn is not None:
                            pair_i, pair_m = update_fn(np.asarray(pos), box=pbc_box_nl)
                            _pbc_state["pair_idx"] = pair_i
                            _pbc_state["pair_mask"] = pair_m

                    def _fire_log(stage_idx, dt_ps, i, n_tot, energy, max_force):
                        c.print(
                            f"  [dim]dt={dt_ps:.1e} stage {stage_idx} {i}/{n_tot}[/dim]: "
                            f"E={energy if energy is not None else float('nan'):.6f} eV, "
                            f"max|F|={max_force:.6f}"
                        )

                    minimized_pos, best_fire_max_f, fire_info = run_jaxmd_fire_with_dt_backoff(
                        force_fn=wrapped_force_fn,
                        shift_fn=fire_shift_fn,
                        positions=initial_pos,
                        masses=Si_mass,
                        n_steps=int(n_fire),
                        dt_schedule=dt_sched,
                        nl_refresh_fn=_fire_nl_refresh if (use_pbc and update_fn is not None) else None,
                        energy_fn=wrapped_energy_fn,
                        log_fn=_fire_log,
                    )
                    minimized_pos, best_fire_max_f, fire_info = (
                        maybe_fire_monomer_template_rebuild_retry(
                            positions=minimized_pos,
                            best_max_f=best_fire_max_f,
                            fire_info=fire_info,
                            force_fn=wrapped_force_fn,
                            energy_fn=wrapped_energy_fn,
                            shift_fn=fire_shift_fn,
                            masses=Si_mass,
                            monomer_offsets=monomer_offsets,
                            atomic_numbers=atomic_numbers,
                            nl_refresh_fn=(
                                _fire_nl_refresh if (use_pbc and update_fn is not None) else None
                            ),
                            log_fn=_fire_log,
                            console=c,
                        )
                    )
                    fire_positions.append(minimized_pos)
                    if best_fire_max_f < fire_info["start_max_f"] - 1.0e-6:
                        blow_note = (
                            "; stage(s) aborted on force blow-up"
                            if fire_info.get("blew_up")
                            else ""
                        )
                        rebuild_note = (
                            f"; template rebuild n={fire_info['template_rebuild']['n_rebuilt']}"
                            if fire_info.get("template_rebuild")
                            else ""
                        )
                        c.print(
                            Panel(
                                f"FIRE improved max|F| "
                                f"{fire_info['start_max_f']:.4f} → {best_fire_max_f:.4f} "
                                f"(dt_final={fire_info.get('dt_final_ps')} ps, "
                                f"stages={len(fire_info['stages'])}"
                                f"{blow_note}{rebuild_note})",
                                title="[bold green]JAX-MD Minimization[/bold green]",
                                border_style="green",
                            )
                        )
                    else:
                        # First FIRE already used molecular wrap under PBC — PBC FIRE
                        # would repeat the same no-op backoff (unless a blow-up left
                        # the geometry still hard — then let PBC FIRE try).
                        skip_redundant_pbc_fire = should_skip_redundant_pbc_fire(
                            first_fire_steps=int(NMIN),
                            first_fire_ran_without_improvement=not bool(
                                fire_info.get("blew_up")
                            ),
                            use_pbc=bool(use_pbc),
                        )
                        c.print(
                            Panel(
                                f"FIRE did not improve max|F| "
                                f"(kept {best_fire_max_f:.4f}; tried dt={list(dt_sched)} ps"
                                f"{'; blew up' if fire_info.get('blew_up') else ''}). "
                                "ASE/CHARMM geometry already soft — continuing.",
                                title="[bold]JAX-MD Minimization[/bold]",
                                border_style="yellow",
                            )
                        )
        res_overlap = _check_overlap(
            minimized_pos,
            atoms.get_cell()[:] if use_pbc else None,
            "after JAX-MD first minimization",
        )
        if res_overlap is not None:
            minimized_pos = jnp.asarray(res_overlap, dtype=jnp.float32)
        # save pdb (wrap by monomer when PBC so molecules stay intact)
        min_pdb_path = Path(f"{args.output_prefix}_minimized.pdb")
        min_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        if use_pbc:
            _cell_for_pdb = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            pos_wrapped = _wrap_monomers(jnp.asarray(minimized_pos), _cell_for_pdb)
            atoms.set_positions(np.asarray(jax.device_get(pos_wrapped)))
        ase_io.write(str(min_pdb_path), atoms)

        # ========================================================================
        # PBC MINIMIZATION (only when PBC enabled, i.e. cell is set)
        # ========================================================================
        pbc_fire_positions = []
        if not use_pbc:
            md_pos = minimized_pos
            c.print(Panel("No cell: skipping PBC minimization", title="[bold]PBC Minimization[/bold]", border_style="yellow"))
            atoms.set_positions(np.asarray(md_pos))
        else:
            NMIN_PBC = getattr(args, "jaxmd_pbc_minimize_steps", 1000)
            # Molecular shift: wrap by monomer after each step so monomers stay intact.
            # space.periodic wraps atoms individually → monomers break across boundaries.
            _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)

            def shift_molecular(R, dR, **kwargs):
                return _wrap_monomers(R + dR, _cell_jax)

            # Start from wrapped positions so we're in the cell (first min can drift)
            if pbc_map_fn is not None:
                pbc_start_pos = pbc_map_fn(minimized_pos)
            else:
                pbc_start_pos = _wrap_monomers(jnp.asarray(minimized_pos), _cell_jax)
            if update_fn is not None:
                pbc_pair_idx, pbc_pair_mask = update_fn(
                    np.asarray(pbc_start_pos), box=pbc_box_nl
                )
                _pbc_state["pair_idx"] = pbc_pair_idx
                _pbc_state["pair_mask"] = pbc_pair_mask

            f0_pbc = (
                float(jnp.abs(wrapped_force_fn(pbc_start_pos)).max())
                if jnp.all(jnp.isfinite(pbc_start_pos))
                else float("inf")
            )
            if NMIN_PBC <= 0 or jnp.any(~jnp.isfinite(pbc_start_pos)):
                reason = "0 steps requested" if NMIN_PBC <= 0 else "no valid start position"
                print(f"Skipping PBC minimization ({reason})")
                md_pos = pbc_start_pos if jnp.all(jnp.isfinite(pbc_start_pos)) else minimized_pos
            elif skip_redundant_pbc_fire or should_skip_jaxmd_fire(
                f0_pbc, skip_below_eVA=fire_skip_thr
            ):
                md_pos = pbc_start_pos
                why = (
                    "first FIRE already covered molecular-wrap / soft start"
                    if skip_redundant_pbc_fire
                    else (
                        f"start max|F|={f0_pbc:.4f} ≤ skip gate {fire_skip_thr:.4f} eV/Å"
                    )
                )
                c.print(
                    Panel(
                        f"Skipping PBC FIRE ({why}).",
                        title="[bold]PBC Minimization[/bold]",
                        border_style="yellow",
                    )
                )
            else:
                n_pbc_fire = resolve_jaxmd_fire_stage_steps(int(NMIN_PBC), f0_pbc)
                dt0_pbc = resolve_jaxmd_fire_dt_start_ps(f0_pbc)
                dt_sched_pbc = jaxmd_fire_dt_backoff_schedule(dt0_pbc)
                c.print(
                    Panel(
                        f"PBC FIRE minimization ({n_pbc_fire} steps/stage; "
                        f"dt schedule={[f'{d:.1e}' for d in dt_sched_pbc]} ps)",
                        title="[bold cyan]PBC Minimization[/bold cyan]",
                        border_style="cyan",
                    )
                )

                def _pbc_nl_refresh(pos):
                    if update_fn is not None:
                        pair_i, pair_m = update_fn(np.asarray(pos), box=pbc_box_nl)
                        _pbc_state["pair_idx"] = pair_i
                        _pbc_state["pair_mask"] = pair_m

                def _pbc_log(stage_idx, dt_ps, i, n_tot, energy, max_force):
                    c.print(
                        f"  [dim]dt={dt_ps:.1e} stage {stage_idx} {i}/{n_tot}[/dim]: "
                        f"E={energy if energy is not None else float('nan'):.6f} eV, "
                        f"max|F|={max_force:.6f}"
                    )

                md_pos, best_pbc_max_f, pbc_info = run_jaxmd_fire_with_dt_backoff(
                    force_fn=wrapped_force_fn,
                    shift_fn=shift_molecular,
                    positions=pbc_start_pos,
                    masses=Si_mass,
                    n_steps=int(n_pbc_fire),
                    dt_schedule=dt_sched_pbc,
                    nl_refresh_fn=_pbc_nl_refresh if update_fn is not None else None,
                    energy_fn=wrapped_energy_fn,
                    log_fn=_pbc_log,
                )
                md_pos, best_pbc_max_f, pbc_info = maybe_fire_monomer_template_rebuild_retry(
                    positions=md_pos,
                    best_max_f=best_pbc_max_f,
                    fire_info=pbc_info,
                    force_fn=wrapped_force_fn,
                    energy_fn=wrapped_energy_fn,
                    shift_fn=shift_molecular,
                    masses=Si_mass,
                    monomer_offsets=monomer_offsets,
                    atomic_numbers=atomic_numbers,
                    nl_refresh_fn=_pbc_nl_refresh if update_fn is not None else None,
                    log_fn=_pbc_log,
                    console=c,
                )
                pbc_fire_positions.append(md_pos)
                if best_pbc_max_f < pbc_info["start_max_f"] - 1.0e-6:
                    c.print(
                        Panel(
                            f"PBC FIRE improved max|F| "
                            f"{pbc_info['start_max_f']:.4f} → {best_pbc_max_f:.4f}"
                            f"{' (after template rebuild)' if pbc_info.get('template_rebuild') else ''}",
                            title="[bold green]PBC Minimization[/bold green]",
                            border_style="green",
                        )
                    )
                else:
                    c.print(
                        Panel(
                            f"PBC FIRE kept best max|F|={best_pbc_max_f:.4f} "
                            f"(start {pbc_info['start_max_f']:.4f})",
                            title="[bold]PBC Minimization[/bold]",
                            border_style="yellow",
                        )
                    )

            # Save PBC minimized structure (md_pos already wrapped by monomer)
            atoms.set_positions(np.asarray(md_pos))
            res_after_pbc = _check_overlap(md_pos, atoms.get_cell()[:], "after JAX-MD PBC minimization")
            if res_after_pbc is not None:
                md_pos = jnp.asarray(res_after_pbc, dtype=jnp.float32)
                atoms.set_positions(np.asarray(res_after_pbc, dtype=float))
            pbc_pdb_path = Path(f"{args.output_prefix}_pbc_minimized.pdb")
            pbc_pdb_path.parent.mkdir(parents=True, exist_ok=True)
            ase_io.write(str(pbc_pdb_path), atoms)
            c.print(Panel(f"Complete. Final energy: {float(wrapped_energy_fn(md_pos)):.6f} eV", title="[bold green]PBC Minimization[/bold green]", border_style="green"))

        # Use last valid positions if minimization produced NaN
        if jnp.any(~jnp.isfinite(md_pos)) and pbc_fire_positions:
            md_pos = pbc_fire_positions[-1]
            c.print(Panel("NaN in PBC minimization; using last valid position from PBC", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
        if jnp.any(~jnp.isfinite(md_pos)) and fire_positions:
            md_pos = pbc_map_fn(fire_positions[-1]) if (use_pbc and pbc_map_fn) else fire_positions[-1]
            c.print(Panel("Using last valid position from first minimization", title="[bold yellow]Warning[/bold yellow]", border_style="yellow"))
        if jnp.any(~jnp.isfinite(md_pos)):
            c.print(Panel(f"No valid positions for {args.ensemble.upper()}; skipping JAX-MD", title="[bold red]Error[/bold red]", border_style="red"))
            run_sim.last_status = "error"
            run_sim.last_error = "No valid positions for JAX-MD"
            return 0, jnp.array([]).reshape(0, len(md_pos), 3), None
        res_pre = _check_overlap(md_pos, atoms.get_cell()[:] if use_pbc else None, "before JAX-MD dynamics")
        if res_pre is not None:
            md_pos = jnp.asarray(res_pre, dtype=jnp.float32)
            atoms.set_positions(np.asarray(res_pre, dtype=float))

        current_neighbors = None
        if use_pbc and update_fn is not None:
            current_neighbors = (_pbc_state["pair_idx"], _pbc_state["pair_mask"])

        if args.ensemble == "npt" and use_pbc:
            # NPT: positions in fractional coords; wrap md_pos into cell first, then convert to fractional
            box_curr = box_npt
            _cell_jax = jnp.asarray(atoms.get_cell()[:], dtype=jnp.float32)
            md_pos_wrapped = _wrap_monomers(jnp.asarray(md_pos), _cell_jax)
            md_pos_frac = as_jaxmd_dtype(md_pos_wrapped / float(args.cell))  # cubic: frac = R / L
            # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
            box_nl = np.array([float(args.cell)] * 3, dtype=np.float64)
            pair_idx, pair_mask = update_fn(md_pos_frac, box=box_nl)
            state = init_fn(
                key, md_pos_frac, box=box_curr,
                neighbor=(pair_idx, pair_mask), kT=kT, mass=Si_mass
            )
            npt_pair_idx, npt_pair_mask = pair_idx, pair_mask
            current_neighbors = (npt_pair_idx, npt_pair_mask)
            npt_pressure = pressure  # Use same pressure as NPT block (handles --pressure 0)
        elif args.ensemble == "nvt":
            state = init_fn(key, as_jaxmd_dtype(md_pos), mass=Si_mass)
            npt_pair_idx, npt_pair_mask = None, None
            npt_pressure = None
        else:
            state = init_fn(key, as_jaxmd_dtype(md_pos), kT, mass=Si_mass)
            npt_pair_idx, npt_pair_mask = None, None
            npt_pressure = None
        if initial_velocities is not None:
            state = state.set(
                momentum=as_jaxmd_dtype(
                    Si_mass[:, None] * as_jaxmd_dtype(initial_velocities)
                )
            )
        state = normalize_jaxmd_state(state)
        if initial_velocities is not None:
            # Caller may pass handoff OR ASE Maxwell–Boltzmann metal velocities.
            mom_title = "Using provided initial velocities (JAX-MD metal units)"
        else:
            mom_title = f"Maxwell–Boltzmann momentum at {T} K (integrator)"
        c.print(Panel(mom_title, title="[bold]JAX-MD[/bold]", border_style="green"))
        nhc_positions = []
        nhc_boxes = []  # NPT: box at each record step (for frac→real when saving)

        # get energy of initial state
        if is_npt and npt_pair_idx is not None:
            box_curr = simulate.npt_box(state)
            energy_initial = float(npt_energy_fn(state.position, box=box_curr, neighbor=(npt_pair_idx, npt_pair_mask)))
        else:
            pair_idx, pair_mask = current_neighbors if current_neighbors is not None else (None, None)
            out_init = jax_md_eval_fn(
                state.position,
                mm_pair_idx=pair_idx,
                mm_pair_mask=pair_mask,
                box=_pbc_state["box"],
            )
            energy_initial = float(out_init.energy)
            print_flat_bottom_summary(
                out_init,
                flat_bottom_radius=flat_bottom_radius,
                flat_bottom_k=flat_bottom_k,
                flat_bottom_mode=flat_bottom_mode,
                label="MD start (post-FIRE)",
                console=c,
            )
        # Debug: forces from calculator (used by NVE; jax.grad gives NaN)
        if is_npt and npt_pair_idx is not None:
            box_curr = simulate.npt_box(state)
            real_pos = space.transform(box_curr, state.position)
            forces_jax = jax_md_force_fn(
                real_pos,
                mm_pair_idx=npt_pair_idx,
                mm_pair_mask=npt_pair_mask,
                box=box_curr,
            )
        else:
            forces_jax = wrapped_force_fn(state.position, neighbor=current_neighbors)
        print_forces_summary(np.asarray(forces_jax), energy_eV=energy_initial, console=c)
        if args.ensemble == "nve":
            max_f_start = float(jnp.max(jnp.linalg.norm(forces_jax, axis=-1)))
            n_atoms_gate = int(np.asarray(forces_jax).shape[0])
            fmax_gate, fmax_scale = resolve_nve_max_f_start_gate_eVA(
                getattr(args, "nve_max_f_start_eVA", NVE_MAX_F_START_BASE_EVA),
                n_atoms=n_atoms_gate,
            )
            if fmax_gate > 0.0:
                base_cfg = float(
                    getattr(args, "nve_max_f_start_eVA", NVE_MAX_F_START_BASE_EVA) or 0.0
                )
                c.print(
                    Panel(
                        f"NVE start max|F| gate: {fmax_gate:.4f} eV/Å "
                        f"(base {base_cfg:.4f} × size scale {fmax_scale:.3f} "
                        f"for N={n_atoms_gate}, ref N={NVE_MAX_F_START_N_ATOMS_REF}; "
                        f"live max|F|={max_f_start:.4f})",
                        title="[bold]NVE start force gate[/bold]",
                        border_style="cyan",
                    )
                )
            if fmax_gate > 0.0 and max_f_start > fmax_gate:
                msg = (
                    f"NVE refused: post-FIRE max|F|={max_f_start:.4f} eV/Å "
                    f"> size-scaled gate {fmax_gate:.4f} eV/Å "
                    f"(base {float(getattr(args, 'nve_max_f_start_eVA', NVE_MAX_F_START_BASE_EVA) or 0.0):.4f} "
                    f"× {fmax_scale:.3f} for N={n_atoms_gate}). "
                    "Improve minimization / packing before microcanonical dynamics, "
                    "raise --nve-max-f-start-eVA, or use a value <=0 to disable."
                )
                c.print(
                    Panel(
                        msg,
                        title="[bold red]NVE preflight failed[/bold red]",
                        border_style="red",
                    )
                )
                run_sim.last_status = "error"
                run_sim.last_error = msg
                pos0 = np.asarray(jax.device_get(state.position), dtype=float)
                return 0, np.stack([pos0]), None
            # float32 energy differences are too coarse for force–energy FD and
            # to reduce integration error on stiff hybrid potentials. Conservation
            # must be established from each run's energy-drift receipt.
            x64_on = bool(jax.config.read("jax_enable_x64"))
            if (not x64_on) or _JAXMD_DTYPE != jnp.float64:
                msg = (
                    "NVE requires JAX float64. Export JAX_ENABLE_X64=1 "
                    "*before* starting Python (and use --ml-compute-dtype float64 "
                    "or MMML_ML_DTYPE=float64). "
                    f"Current: jax_enable_x64={x64_on}, "
                    f"ml_dtype={_JAXMD_DTYPE}."
                )
                c.print(
                    Panel(
                        msg,
                        title="[bold red]NVE preflight failed[/bold red]",
                        border_style="red",
                    )
                )
                run_sim.last_status = "error"
                run_sim.last_error = msg
                pos0 = np.asarray(jax.device_get(state.position), dtype=float)
                return 0, np.stack([pos0]), None
            # NVE is only meaningful when the explicit calculator forces are
            # the negative derivative of the reported potential energy.  The
            # hybrid calculator cannot currently be differentiated end-to-end,
            # so verify that contract numerically before integrating.
            fd_tol = float(
                getattr(args, "nve_force_energy_relative_tolerance", 0.20)
                or 0.0
            )
            if fd_tol > 0.0:
                fd_eps = float(
                    getattr(args, "nve_force_energy_epsilon_A", 0.01) or 0.01
                )
                rescue_enabled = bool(
                    getattr(args, "nve_force_energy_rescue", True)
                )
                rescue_fire_steps = int(
                    getattr(args, "nve_force_energy_rescue_fire_steps", 50) or 0
                )
                ncoord = int(np.prod(state.position.shape))
                direction_np = np.sin(
                    np.arange(1, ncoord + 1, dtype=np.float64)
                ).reshape(tuple(state.position.shape))
                direction_np /= max(float(np.linalg.norm(direction_np)), 1.0e-12)
                direction = as_jaxmd_dtype(direction_np)

                _mm_charge_mode = str(
                    getattr(args, "mm_charge_mode", None) or "fixed"
                ).strip().lower()
                # Q⁰ / latent* MM charges depend on R, but MM forces are
                # Hellmann–Feynman (∂E_MM/∂R|_q) — same as hybrid_forward training.
                # FD of E(R, q(R)) therefore disagrees with F unless q is frozen.
                try:
                    from mmml.models.mm_charge_mode import mm_charge_mode_needs_q_ml

                    _freeze_q_for_fd = bool(
                        getattr(args, "include_mm", True)
                    ) and mm_charge_mode_needs_q_ml(_mm_charge_mode)
                except Exception:
                    _freeze_q_for_fd = _mm_charge_mode in {
                        "q0",
                        "latent",
                        "q1",
                        "fixed_plus_latent",
                        "latent_dynamic",
                    }
                if getattr(args, "nve_force_energy_freeze_charges", None) is not None:
                    _freeze_q_for_fd = bool(
                        getattr(args, "nve_force_energy_freeze_charges")
                    )

                def _hybrid_fd_check(pos, forces, neighbors, *, freeze_mm_charges: bool):
                    pair_i = neighbors[0] if neighbors is not None else None
                    pair_m = neighbors[1] if neighbors is not None else None
                    box_fd = _pbc_state["box"]
                    pos0 = as_jaxmd_dtype(pos)
                    if freeze_mm_charges:
                        out0 = spherical_cutoff_calculator(
                            atomic_numbers=atomic_numbers,
                            positions=pos0,
                            n_monomers=n_monomers,
                            cutoff_params=CUTOFF_PARAMS,
                            doML=True,
                            doMM=True,
                            doML_dimer=not getattr(args, "skip_ml_dimers", False),
                            debug=False,
                            mm_pair_idx=pair_i,
                            mm_pair_mask=pair_m,
                            box=box_fd,
                        )
                        q_freeze = jax.lax.stop_gradient(out0.mm_charges)

                        def _e_hf(p):
                            out = spherical_cutoff_calculator(
                                atomic_numbers=atomic_numbers,
                                positions=as_jaxmd_dtype(p),
                                n_monomers=n_monomers,
                                cutoff_params=CUTOFF_PARAMS,
                                doML=True,
                                doMM=True,
                                doML_dimer=not getattr(args, "skip_ml_dimers", False),
                                debug=False,
                                mm_pair_idx=pair_i,
                                mm_pair_mask=pair_m,
                                box=box_fd,
                                use_mm_charges_override=True,
                                mm_charges_override=q_freeze,
                            )
                            return out.energy.reshape(-1)[0]

                        e_plus = float(_e_hf(pos0 + fd_eps * direction))
                        e_minus = float(_e_hf(pos0 - fd_eps * direction))
                    else:
                        e_plus = float(
                            wrapped_energy_fn(
                                pos + fd_eps * direction,
                                neighbor=neighbors,
                            )
                        )
                        e_minus = float(
                            wrapped_energy_fn(
                                pos - fd_eps * direction,
                                neighbor=neighbors,
                            )
                        )
                    projected = float(jnp.sum(forces * direction))
                    slope, relerr = directional_force_energy_error(
                        e_plus,
                        e_minus,
                        fd_eps,
                        projected,
                    )
                    return slope, relerr, projected

                def _ml_only_fd_check(pos):
                    pos0_jax = as_jaxmd_dtype(pos)
                    box_fd = _pbc_state["box"]

                    def _ml_only_energy(p):
                        out = spherical_cutoff_calculator(
                            atomic_numbers=atomic_numbers,
                            positions=as_jaxmd_dtype(p),
                            n_monomers=n_monomers,
                            cutoff_params=CUTOFF_PARAMS,
                            doML=True,
                            doMM=False,
                            doML_dimer=not getattr(args, "skip_ml_dimers", False),
                            debug=False,
                            box=box_fd,
                        )
                        return out.energy.reshape(-1)[0]

                    ml_out0 = spherical_cutoff_calculator(
                        atomic_numbers=atomic_numbers,
                        positions=pos0_jax,
                        n_monomers=n_monomers,
                        cutoff_params=CUTOFF_PARAMS,
                        doML=True,
                        doMM=False,
                        doML_dimer=not getattr(args, "skip_ml_dimers", False),
                        debug=False,
                        box=box_fd,
                    )
                    ml_forces = as_jaxmd_dtype(ml_out0.forces)
                    ml_e_plus = float(_ml_only_energy(pos0_jax + fd_eps * direction))
                    ml_e_minus = float(_ml_only_energy(pos0_jax - fd_eps * direction))
                    ml_proj = float(jnp.sum(ml_forces * direction))
                    return directional_force_energy_error(
                        ml_e_plus,
                        ml_e_minus,
                        fd_eps,
                        ml_proj,
                    ) + (ml_proj,)

                fd_slope, fd_relerr, projected_force = _hybrid_fd_check(
                    state.position,
                    forces_jax,
                    current_neighbors,
                    freeze_mm_charges=_freeze_q_for_fd,
                )
                ml_only_relerr: float | None = None
                ml_only_slope: float | None = None
                do_ml_ablation = bool(
                    getattr(args, "include_mm", True)
                ) and bool(
                    getattr(args, "nve_force_energy_ml_only_diagnose", True)
                )
                if do_ml_ablation:
                    ml_only_slope, ml_only_relerr, ml_proj = _ml_only_fd_check(
                        state.position
                    )
                else:
                    ml_proj = None
                _fd_title = "NVE force–energy preflight"
                if _freeze_q_for_fd:
                    _fd_title += " (Hellmann–Feynman: q_MM frozen)"
                c.print(
                    Panel(
                        f"directional FD dE/ds={fd_slope:.6f} eV/Å, "
                        f"-F·d={-projected_force:.6f} eV/Å, "
                        f"relative error={fd_relerr:.4f} (tol={fd_tol:.4f})",
                        title=f"[bold]{_fd_title}[/bold]",
                        border_style="cyan",
                    )
                )
                def _ablation_verdict(hyb_err, ml_err):
                    return nve_force_energy_ablation_verdict(
                        hyb_err,
                        ml_err,
                        fd_tol,
                        mm_charge_mode=_mm_charge_mode,
                        used_frozen_mm_charges=_freeze_q_for_fd,
                    )

                if do_ml_ablation and ml_only_relerr is not None:
                    verdict = _ablation_verdict(fd_relerr, ml_only_relerr)
                    c.print(
                        Panel(
                            f"ML-only (doMM=False): dE/ds={ml_only_slope:.6f} eV/Å, "
                            f"-F·d={-ml_proj:.6f} eV/Å, "
                            f"relative error={ml_only_relerr:.4f}\n"
                            f"hybrid (doMM={bool(getattr(args, 'include_mm', True))}): "
                            f"relative error={fd_relerr:.4f}\n"
                            f"verdict: {verdict}",
                            title="[bold]NVE force–energy ML-only ablation[/bold]",
                            border_style="magenta",
                        )
                    )

                rescue_attempted = False
                if nve_force_energy_should_attempt_rescue(
                    fd_relerr,
                    fd_tol,
                    rescue_enabled=rescue_enabled,
                    rescue_already_attempted=False,
                ):
                    rescue_attempted = True
                    c.print(
                        Panel(
                            "Hybrid force–energy FD failed — attempting rescue: "
                            "force MM neighbor-list rebuild"
                            + (
                                f" + short jax-md FIRE ({rescue_fire_steps} steps)"
                                if rescue_fire_steps > 0
                                else ""
                            )
                            + ", then re-check.",
                            title="[bold yellow]NVE preflight rescue[/bold yellow]",
                            border_style="yellow",
                        )
                    )
                    pos_rescue = state.position
                    if use_pbc and update_fn is not None:
                        try:
                            pair_i, pair_m = update_fn(
                                _nl_update_positions(pos_rescue),
                                box=pbc_box_nl,
                                force_rebuild=True,
                            )
                        except TypeError:
                            # Older update_fn closures without force_rebuild kwarg.
                            pair_i, pair_m = update_fn(
                                _nl_update_positions(pos_rescue),
                                box=pbc_box_nl,
                            )
                        _pbc_state["pair_idx"] = pair_i
                        _pbc_state["pair_mask"] = pair_m
                        current_neighbors = (pair_i, pair_m)
                    if rescue_fire_steps > 0:
                        def _rescue_nl_refresh(pos):
                            if use_pbc and update_fn is not None:
                                try:
                                    pair_i, pair_m = update_fn(
                                        _nl_update_positions(pos),
                                        box=pbc_box_nl,
                                        force_rebuild=True,
                                    )
                                except TypeError:
                                    pair_i, pair_m = update_fn(
                                        _nl_update_positions(pos),
                                        box=pbc_box_nl,
                                    )
                                _pbc_state["pair_idx"] = pair_i
                                _pbc_state["pair_mask"] = pair_m

                        rescue_shift_fn = shift_min
                        if use_pbc:
                            _cell_rescue = jnp.asarray(
                                atoms.get_cell()[:], dtype=jnp.float32
                            )

                            def rescue_shift_fn(pos, dR, **kwargs):
                                return _wrap_monomers(pos + dR, _cell_rescue)

                        dt_rescue = resolve_jaxmd_fire_dt_start_ps(
                            float(jnp.max(jnp.linalg.norm(forces_jax, axis=-1)))
                        )
                        pos_rescue, best_f, fire_info = run_jaxmd_fire_with_dt_backoff(
                            force_fn=wrapped_force_fn,
                            shift_fn=rescue_shift_fn,
                            positions=pos_rescue,
                            masses=Si_mass,
                            n_steps=int(rescue_fire_steps),
                            dt_schedule=(dt_rescue,),
                            nl_refresh_fn=(
                                _rescue_nl_refresh
                                if (use_pbc and update_fn is not None)
                                else None
                            ),
                            energy_fn=wrapped_energy_fn,
                        )
                        c.print(
                            Panel(
                                f"Rescue FIRE: max|F| "
                                f"{fire_info['start_max_f']:.4f} → {best_f:.4f} eV/Å "
                                f"({rescue_fire_steps} steps)",
                                title="[bold]NVE preflight rescue[/bold]",
                                border_style="yellow",
                            )
                        )
                    state = state.set(position=as_jaxmd_dtype(pos_rescue))
                    state = normalize_jaxmd_state(state)
                    if use_pbc and update_fn is not None:
                        try:
                            pair_i, pair_m = update_fn(
                                _nl_update_positions(state.position),
                                box=pbc_box_nl,
                                force_rebuild=True,
                            )
                        except TypeError:
                            pair_i, pair_m = update_fn(
                                _nl_update_positions(state.position),
                                box=pbc_box_nl,
                            )
                        _pbc_state["pair_idx"] = pair_i
                        _pbc_state["pair_mask"] = pair_m
                        current_neighbors = (pair_i, pair_m)
                    forces_jax = wrapped_force_fn(
                        state.position, neighbor=current_neighbors
                    )
                    energy_initial = float(
                        wrapped_energy_fn(
                            state.position, neighbor=current_neighbors
                        )
                    )
                    print_forces_summary(
                        np.asarray(forces_jax),
                        energy_eV=energy_initial,
                        console=c,
                    )
                    fd_slope, fd_relerr, projected_force = _hybrid_fd_check(
                        state.position,
                        forces_jax,
                        current_neighbors,
                        freeze_mm_charges=_freeze_q_for_fd,
                    )
                    if do_ml_ablation:
                        ml_only_slope, ml_only_relerr, ml_proj = _ml_only_fd_check(
                            state.position
                        )
                    _fd_title_rescue = "NVE force–energy preflight (after rescue)"
                    if _freeze_q_for_fd:
                        _fd_title_rescue += " (Hellmann–Feynman: q_MM frozen)"
                    c.print(
                        Panel(
                            f"directional FD dE/ds={fd_slope:.6f} eV/Å, "
                            f"-F·d={-projected_force:.6f} eV/Å, "
                            f"relative error={fd_relerr:.4f} (tol={fd_tol:.4f})",
                            title=f"[bold]{_fd_title_rescue}[/bold]",
                            border_style="cyan",
                        )
                    )
                    if do_ml_ablation and ml_only_relerr is not None:
                        verdict = _ablation_verdict(fd_relerr, ml_only_relerr)
                        c.print(
                            Panel(
                                f"ML-only (doMM=False): dE/ds={ml_only_slope:.6f} eV/Å, "
                                f"-F·d={-ml_proj:.6f} eV/Å, "
                                f"relative error={ml_only_relerr:.4f}\n"
                                f"hybrid (doMM={bool(getattr(args, 'include_mm', True))}): "
                                f"relative error={fd_relerr:.4f}\n"
                                f"verdict: {verdict}",
                                title="[bold]NVE force–energy ML-only ablation[/bold]",
                                border_style="magenta",
                            )
                        )
                    if np.isfinite(fd_relerr) and fd_relerr <= fd_tol:
                        c.print(
                            Panel(
                                f"Rescue succeeded: hybrid relative error "
                                f"{fd_relerr:.4f} ≤ {fd_tol:.4f}; continuing NVE.",
                                title="[bold green]NVE preflight rescue[/bold green]",
                                border_style="green",
                            )
                        )

                if not np.isfinite(fd_relerr) or fd_relerr > fd_tol:
                    msg = (
                        "NVE force–energy consistency failed: "
                        f"finite-difference dE/ds={fd_slope:.6f} eV/Å, "
                        f"-F·d={-projected_force:.6f} eV/Å, relative error "
                        f"{fd_relerr:.3f} > {fd_tol:.3f}. The hybrid force is "
                        "non-conservative for this configuration; NVE would heat "
                        "and drift."
                    )
                    if rescue_attempted:
                        msg += " Rescue (NL rebuild + short FIRE) did not recover."
                    if ml_only_relerr is not None:
                        msg += (
                            f" ML-only relerr={ml_only_relerr:.3f}"
                            + (
                                f" (dE/ds={ml_only_slope:.6f})"
                                if ml_only_slope is not None
                                else ""
                            )
                            + f". {_ablation_verdict(fd_relerr, ml_only_relerr)}"
                        )
                    c.print(
                        Panel(
                            msg,
                            title="[bold red]NVE preflight failed[/bold red]",
                            border_style="red",
                        )
                    )
                    run_sim.last_status = "error"
                    run_sim.last_error = msg
                    pos0 = np.asarray(jax.device_get(state.position), dtype=float)
                    return 0, np.stack([pos0]), None
        # velocity = momentum / mass; position update = R + dt * v (half-step in VV)
        vel = state.momentum / state.mass
        disp_first = dt * vel
        t_vel = Table(title="First-step kinematics")
        t_vel.add_column("Property", style="cyan")
        t_vel.add_column("Value", style="white")
        t_vel.add_row("velocity sample [0]", str(np.asarray(vel[0])))
        t_vel.add_row("disp dt*v [0]", str(np.asarray(disp_first[0])))
        t_vel.add_row("max|disp|", f"{float(jnp.max(jnp.abs(disp_first))):.6f}")
        c.print(Panel(t_vel, title="[bold]JAX-MD First Step[/bold]", border_style="blue"))

        # ========================================================================
        # NPT DIAGNOSTIC TESTS (--npt-diagnose)
        # ========================================================================
        if is_npt and npt_pair_idx is not None and getattr(args, "npt_diagnose", False):
            _run_npt_diagnostics(
                state=state,
                npt_energy_fn=npt_energy_fn,
                jax_md_force_fn=jax_md_force_fn,
                apply_fn=apply_fn,
                shift=shift,
                space=space,
                simulate=simulate,
                quantity=quantity,
                npt_pair_idx=npt_pair_idx,
                npt_pair_mask=npt_pair_mask,
                npt_pressure=npt_pressure,
                unit=unit,
                dt=dt,
                kT=kT,
                grad=grad,
            )

        # Warm up jitted integrator before timed/diagnostic first step.
        if is_npt and npt_pair_idx is not None:
            _warm_state = apply_fn(
                state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure
            )
        else:
            _warm_state = apply_fn(state, neighbor=current_neighbors)
        block_jax_values(_warm_state.position, _warm_state.momentum)

        # Single-step diagnostic: catch NaN on first step (common with wrong mass/units)
        if is_npt and npt_pair_idx is not None:
            state_one = apply_fn(state, neighbor=(npt_pair_idx, npt_pair_mask), pressure=npt_pressure)
        else:
            state_one = apply_fn(state, neighbor=current_neighbors)
        if not jnp.all(jnp.isfinite(state_one.position)):
            t_err = Table(title="First step NaN")
            t_err.add_column("Check", style="red")
            t_err.add_column("Value", style="white")
            t_err.add_row("mass shape", str(state.mass.shape))
            t_err.add_row("mass min/max", f"{float(jnp.min(state.mass)):.4f} / {float(jnp.max(state.mass)):.4f}")
            c.print(Panel(t_err, title="[bold red]ERROR: First step produced NaN positions[/bold red]\nCheck: mass in amu, dt in ps, energy_fn returns eV.", border_style="red"))
            pos_out = space.transform(simulate.npt_box(state), state.position) if is_npt else state.position
            box_out = [np.asarray(jax.device_get(simulate.npt_box(state)))] if is_npt else None
            run_sim.last_status = "error"
            run_sim.last_error = "First step produced NaN positions"
            return 0, np.stack([np.asarray(jax.device_get(pos_out))]), box_out
        if use_flat_bottom:
            if is_npt and npt_pair_idx is not None:
                box_one = simulate.npt_box(state_one)
                out1 = _eval_at_position(
                    state_one.position,
                    box=box_one,
                    pair_idx=npt_pair_idx,
                    pair_mask=npt_pair_mask,
                )
            else:
                pair_idx, pair_mask = current_neighbors if current_neighbors is not None else (None, None)
                out1 = _eval_at_position(
                    state_one.position,
                    pair_idx=pair_idx,
                    pair_mask=pair_mask,
                )
            e1 = float(out1.energy)
            c.print(
                Panel(
                    f"First step OK: E_pot={e1:.6f} eV, {_fb_dist_hdr}={float(out1.com_dist):.4f}, "
                    f"V_fb={float(out1.flat_bottom_E):.6f} eV",
                    title="[bold green]JAX-MD[/bold green]",
                    border_style="green",
                )
            )
        elif is_npt and npt_pair_idx is not None:
            box_one = simulate.npt_box(state_one)
            e1 = float(npt_energy_fn(state_one.position, box=box_one, neighbor=(npt_pair_idx, npt_pair_mask)))
            c.print(Panel(f"First step OK: E_pot={e1:.6f} eV", title="[bold green]JAX-MD[/bold green]", border_style="green"))
        else:
            e1 = float(wrapped_energy_fn(state_one.position, neighbor=current_neighbors))
            c.print(Panel(f"First step OK: E_pot={e1:.6f} eV", title="[bold green]JAX-MD[/bold green]", border_style="green"))

        nbr_monitor = getattr(args, "nbr_monitor", False)
        if use_pbc:
            c.print(Panel(
                f"{n_monomers} monomer groups: NL rebuild every {steps_per_loop_call} MD steps "
                f"from a wrapped copy (integrator positions stay unwrapped / MIC); "
                f"record every {steps_per_recording}",
                title="[bold]PBC neighbor lists[/bold]",
                border_style="blue",
            ))
        _total_time_ps = total_steps * dt
        c.print(Panel(
            f"Starting {args.ensemble.upper()} simulation | "
            f"total steps: {total_steps:,} | "
            f"total time: {_total_time_ps:.4f} ps | "
            f"dt: {dt_fs:.4f} fs | "
            f"recording every {steps_per_recording} steps ({steps_per_recording * dt:.4f} ps)",
            title="[bold cyan]JAX-MD[/bold cyan]",
            border_style="cyan",
        ))
        _fb_hdr = f"\t{_fb_dist_hdr}\tV_fb (eV)" if use_flat_bottom else ""
        # E_wall reads ~0 in a healthy run. A sustained non-zero value means
        # monomers are riding the short-range wall rather than being caught by it
        # occasionally -- the trajectory is leaving the training data.
        _wall_hdr = "\tE_wall (eV)" if report_wall else ""
        if is_npt:
            hdr = (
                f"\t\tTime (ps) [of {_total_time_ps:.3f} ps, {total_steps:,} steps, dt={dt_fs:.4f} fs]"
                "\tSteps\tE_pot (eV)\tE_tot (eV)\tT (K)\tL (Å)\tV (Å³)\trho (g/cm³)"
                f"\tP_tgt (atm)\tP_meas (atm){_fb_hdr}{_wall_hdr}\tavg(ns/day)"
            )
            if nbr_monitor:
                hdr += "\tn_valid\tcapacity\tfill%"
            c.print(f"[dim]{hdr}[/dim]")
        else:
            c.print(
                f"[dim]\t\tTime (ps) [of {_total_time_ps:.3f} ps, {total_steps:,} steps, dt={dt_fs:.4f} fs]"
                f"\tSteps\tE_pot (eV)\tE_tot (eV)\tT (K){_fb_hdr}{_wall_hdr}\tavg(ns/day)[/dim]"
            )

        # ========================================================================
        # HDF5 REPORTER SETUP
        # ========================================================================
        hdf5_path = Path(f"{args.output_prefix}_{args.ensemble}.h5")
        run_sim.last_hdf5_path = str(hdf5_path)
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        scalar_quantities = ["total_energy", "time_ps"]
        if is_npt:
            scalar_quantities.append("density_g_cm3")
        if nbr_monitor and is_npt:
            scalar_quantities.extend(["nbr_n_valid", "nbr_capacity", "nbr_fill_ratio"])
        if use_flat_bottom:
            scalar_quantities.extend(["com_dist_A", "flat_bottom_E_eV"])
        _mm_charge_mode_attr = getattr(args, "mm_charge_mode", None)
        hdf5_reporter = make_jaxmd_reporter(
            str(hdf5_path),
            n_atoms=len(atoms),
            # Short smokes (e.g. --ps 0.001 with record every 100) can have
            # total_records=0; HDF5 chunks must stay positive.
            buffer_size=max(1, min(100, max(int(total_records), 1))),
            include_positions=True,
            include_velocities=True,
            # Per-frame MM Coulomb charges (Q⁰ / latent / PSF) from ModelOutput.
            include_charges=True,
            scalar_quantities=scalar_quantities,
            attrs={
                "ensemble": args.ensemble,
                "temperature_target": float(T),
                "dt_ps": float(dt),
                "steps_per_recording": int(steps_per_recording),
                "n_atoms": len(atoms),
                "atomic_numbers": np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
                "charges_units": "e",
                "charges_meaning": "per-atom q used for E_MM Coulomb (ModelOutput.mm_charges)",
                **(
                    {"mm_charge_mode": str(_mm_charge_mode_attr)}
                    if _mm_charge_mode_attr is not None
                    else {}
                ),
                **(
                    {
                        "flat_bottom_radius_A": float(flat_bottom_radius),
                        "flat_bottom_k_eV_A2": float(flat_bottom_k),
                        "flat_bottom_mode": flat_bottom_mode,
                    }
                    if use_flat_bottom
                    else {}
                ),
            },
        )

        # ========================================================================
        # MAIN SIMULATION LOOP
        # ========================================================================
        jaxmd_loop_start = time.perf_counter()
        run_status = "complete"
        run_error = None
        rescue_rng = key
        e_tot_ref: float | None = None
        # Abort NVE when |E_tot - E_tot_ref| exceeds this (eV). Non-conservative
        # force bugs otherwise run to multi-1000 K before anyone notices.
        e_tot_drift_abort_eV = float(
            getattr(args, "nve_etot_drift_abort_eV", 0.5) or 0.0
        )
        e_tot_drift_rescue = bool(getattr(args, "nve_etot_drift_rescue", True))
        e_tot_drift_rescue_attempts = int(
            getattr(args, "nve_etot_drift_rescue_attempts", 5) or 0
        )
        e_tot_drift_rescue_fire_steps = int(
            getattr(args, "nve_etot_drift_rescue_fire_steps", 100) or 0
        )
        e_tot_drift_rescue_grace_eV = float(
            getattr(args, "nve_etot_drift_rescue_grace_eV", 2.5) or 0.0
        )
        e_tot_drift_rescue_dt_scale = float(
            getattr(args, "nve_etot_drift_rescue_dt_scale", 0.5) or 0.5
        )
        e_tot_drift_rescue_min_dt_fs = float(
            getattr(args, "nve_etot_drift_rescue_min_dt_fs", 0.05) or 0.05
        )
        e_tot_drift_rescue_count = 0
        e_tot_drift_threshold_eV = e_tot_drift_abort_eV
        last_good_pos = None
        force_progress_print = False
        md_steps_completed = 0
        sim_time_ps = 0.0

        def _state_after_overlap_rescue(
            pos,
            *,
            box_curr=None,
            neighbor=None,
        ):
            """Fresh integrator state at rescued geometry (PyCHARMM-style velocity assign)."""
            nonlocal rescue_rng, npt_pair_idx, npt_pair_mask
            rescue_rng, subkey = jax.random.split(rescue_rng)
            pos_j = as_jaxmd_dtype(pos)
            if is_npt and box_curr is not None:
                neigh = neighbor if neighbor is not None else (npt_pair_idx, npt_pair_mask)
                st = init_fn(
                    subkey,
                    pos_j,
                    box=box_curr,
                    neighbor=neigh,
                    kT=kT,
                    mass=Si_mass,
                )
                npt_pair_idx, npt_pair_mask = neigh
                return normalize_jaxmd_state(st)
            if args.ensemble == "nve":
                return normalize_jaxmd_state(
                    init_fn(subkey, pos_j, kT, mass=Si_mass)
                )
            return normalize_jaxmd_state(init_fn(subkey, pos_j, mass=Si_mass))

        def _rescued_state_energy_finite(st) -> bool:
            if is_npt and npt_pair_idx is not None:
                box_curr = simulate.npt_box(st)
                e = float(
                    npt_energy_fn(
                        st.position,
                        box=box_curr,
                        neighbor=(npt_pair_idx, npt_pair_mask),
                    )
                )
            else:
                e = float(wrapped_energy_fn(st.position, neighbor=current_neighbors))
            return bool(np.isfinite(e))

        try:
            for i in range(total_records):
                steps_done = 0
                while steps_done < steps_per_recording:
                    if is_npt and update_fn is not None:
                        box_curr = simulate.npt_box(state)
                        # Neighbor list with fractional_coordinates expects frac pos and box [L,L,L]
                        box_nl = np.asarray(box_curr)
                        if box_nl.shape == (1,) or box_nl.ndim == 0:
                            L = float(box_nl.reshape(-1)[0])
                            box_nl = np.array([L, L, L], dtype=np.float64)
                        if getattr(args, "debug", False) and (i < 3 or i % 50 == 0) and steps_done == 0:
                            print(f"[nbr] NPT record {i}: updating neighbor list, box L={float(box_nl[0]):.4f}")
                        npt_pair_idx, npt_pair_mask = update_fn(
                            state.position, box=box_nl
                        )
                        current_neighbors = (npt_pair_idx, npt_pair_mask)
                        state = sim(state, neighbor=current_neighbors, pressure=npt_pressure)
                    elif use_pbc and update_fn is not None:
                        # Cell-list binning needs primary-cell coords, but do NOT write
                        # the wrap into integrator state. Whole-monomer ±L jumps make
                        # the hybrid energy discontinuous (~0.1 eV) even though MIC
                        # pair distances are invariant — see mmml_calculator ASE note
                        # ("Do NOT wrap positions during energy/force evaluation").
                        wrapped_for_nl = _wrap_monomers(state.position, _cell_jax)
                        if getattr(args, "debug", False) and (i < 3 or i % 50 == 0) and steps_done == 0:
                            print(f"[nbr] NVT/NVE record {i} (step {steps_done}): updating neighbor list")
                        nvt_neighbors = update_fn(wrapped_for_nl, box=pbc_box_nl)
                        _pbc_state["pair_idx"] = nvt_neighbors[0]
                        _pbc_state["pair_mask"] = nvt_neighbors[1]
                        current_neighbors = nvt_neighbors
                        state = sim(state, neighbor=current_neighbors)
                    else:
                        state = sim(state, neighbor=current_neighbors)
                    steps_done += steps_per_loop_call

                if use_pbc:
                    if is_npt:
                        # NPT: wrap fractional coords to [0,1)
                        box_curr = simulate.npt_box(state)
                        frac_pos = state.position
                        wrapped_frac = frac_pos - jnp.floor(frac_pos)
                        state = state.set(position=as_jaxmd_dtype(wrapped_frac))
                        pos_for_overlap = space.transform(box_curr, state.position)
                        pos_for_overlap = _wrap_monomers(pos_for_overlap, box_curr)
                        rescued = _check_overlap(pos_for_overlap, box_curr, f"JAX-MD dynamics record {i + 1}")
                        if rescued is not None:
                            b_np = np.asarray(jax.device_get(box_curr), dtype=float)
                            new_frac = as_jaxmd_dtype(
                                _real_cartesian_to_fractional(rescued, b_np),
                            )
                            new_frac = new_frac - jnp.floor(new_frac)
                            npt_neighbors = (npt_pair_idx, npt_pair_mask)
                            if update_fn is not None:
                                box_nl = np.asarray(jax.device_get(box_curr))
                                if box_nl.shape == (3, 3):
                                    Ln = float(np.diagonal(box_nl)[:3].mean())
                                    box_nl = np.array([Ln, Ln, Ln], dtype=np.float64)
                                elif box_nl.size >= 3:
                                    box_nl = np.asarray(box_nl, dtype=np.float64).reshape(-1)[:3]
                                npt_neighbors = update_fn(
                                    new_frac, box=box_nl
                                )
                            npt_pair_idx, npt_pair_mask = npt_neighbors
                            current_neighbors = npt_neighbors
                            state = _state_after_overlap_rescue(
                                new_frac,
                                box_curr=box_curr,
                                neighbor=current_neighbors,
                            )
                            if not _rescued_state_energy_finite(state):
                                run_status = "error"
                                run_error = (
                                    f"non-finite MMML energy after overlap rescue "
                                    f"at record {i + 1}"
                                )
                                c.print(Panel(
                                    run_error,
                                    title="[bold red]JAX-MD overlap rescue[/bold red]",
                                    border_style="red",
                                ))
                                break
                    else:
                        # Overlap check on a wrapped copy; keep unwrapped state for energy.
                        pos_for_overlap = _wrap_monomers(state.position, _cell_jax)
                        rescued = _check_overlap(
                            pos_for_overlap, _cell_jax, f"JAX-MD dynamics record {i + 1}"
                        )
                        if rescued is not None:
                            state = _state_after_overlap_rescue(rescued)
                            if update_fn is not None:
                                wrapped_rescue = _wrap_monomers(state.position, _cell_jax)
                                pp_i, pp_m = update_fn(wrapped_rescue, box=pbc_box_nl)
                                _pbc_state["pair_idx"] = pp_i
                                _pbc_state["pair_mask"] = pp_m
                                current_neighbors = (pp_i, pp_m)
                            if not _rescued_state_energy_finite(state):
                                run_status = "error"
                                run_error = (
                                    f"non-finite MMML energy after overlap rescue "
                                    f"at record {i + 1}"
                                )
                                c.print(Panel(
                                    run_error,
                                    title="[bold red]JAX-MD overlap rescue[/bold red]",
                                    border_style="red",
                                ))
                                break
                else:
                    rescued = _check_overlap(state.position, None, f"JAX-MD dynamics record {i + 1}")
                    if rescued is not None:
                        state = _state_after_overlap_rescue(rescued)
                        if not _rescued_state_energy_finite(state):
                            run_status = "error"
                            run_error = (
                                f"non-finite MMML energy after overlap rescue "
                                f"at record {i + 1}"
                            )
                            c.print(Panel(
                                run_error,
                                title="[bold red]JAX-MD overlap rescue[/bold red]",
                                border_style="red",
                            ))
                            break

                # Store current position (NPT: fractional + box for correct real coords at save)
                if is_npt:
                    box_curr = simulate.npt_box(state)
                    nhc_positions.append(state.position)
                    nhc_boxes.append(box_curr)
                else:
                    nhc_positions.append(state.position)

                # Braille viewer: update at each recording block
                if show_frame is not None and atoms_template is not None:
                    steps = (i + 1) * steps_per_recording
                    if is_npt:
                        box_curr = simulate.npt_box(state)
                        pos_real = space.transform(box_curr, state.position)
                        pos_real = _wrap_monomers(pos_real, box_curr)
                    else:
                        pos_real = state.position
                        if use_pbc:
                            pos_real = _wrap_monomers(pos_real, _cell_jax)
                    atoms_template.set_positions(np.asarray(jax.device_get(pos_real)))
                    show_frame(atoms_template, steps, "jaxmd")

                # Energies every record for NVE conservation; print every 10 records.
                # Prefer a full ModelOutput eval so mm_charges land in the HDF5.
                nbr_n_valid = nbr_capacity = nbr_fill_ratio = None
                steps = (i + 1) * steps_per_recording
                time_ps = steps * dt
                T_curr = jax_md.quantity.temperature(
                    momentum=state.momentum,
                    mass=state.mass
                ) / unit['temperature']
                temp = float(T_curr)
                com_dist_report = float("nan")
                e_fb_report = float("nan")
                e_wall_report = float("nan")
                out_dyn = None
                if is_npt and npt_pair_idx is not None:
                    box_curr = simulate.npt_box(state)
                    out_dyn = _eval_at_position(
                        state.position,
                        box=box_curr,
                        pair_idx=npt_pair_idx,
                        pair_mask=npt_pair_mask,
                    )
                else:
                    pair_idx, pair_mask = (
                        current_neighbors
                        if current_neighbors is not None
                        else (None, None)
                    )
                    out_dyn = _eval_at_position(
                        state.position,
                        pair_idx=pair_idx,
                        pair_mask=pair_mask,
                    )
                e_pot = float(out_dyn.energy)
                # Hybrid calculator output omits optional PSF angle restraints;
                # add them so E_pot / E_tot match the forces the integrator sees.
                if _psf_angle_energy_fn is not None:
                    _pos_rep = state.position
                    if is_npt:
                        _pos_rep = space.transform(simulate.npt_box(state), state.position)
                    e_pot += float(_psf_angle_energy_fn(as_jaxmd_dtype(_pos_rep)))
                e_wall_report = float(getattr(out_dyn, "wall_E", float("nan")))
                if use_flat_bottom:
                    com_dist_report = float(out_dyn.com_dist)
                    e_fb_report = float(out_dyn.flat_bottom_E)
                e_kin = float(jax_md.quantity.kinetic_energy(
                    momentum=state.momentum,
                    mass=state.mass
                ))
                e_tot = e_pot + e_kin
                if (
                    not is_npt
                    and args.ensemble == "nve"
                    and e_tot_drift_threshold_eV > 0.0
                    and np.isfinite(e_tot)
                ):
                    if e_tot_ref is None:
                        e_tot_ref = e_tot
                        last_good_pos = state.position
                    elif abs(e_tot - e_tot_ref) > e_tot_drift_threshold_eV:
                        drift = float(e_tot - e_tot_ref)
                        if len(nhc_positions) > 1:
                            nhc_positions = nhc_positions[:-1]
                        can_rescue = (
                            last_good_pos is not None
                            and nve_etot_drift_should_attempt_rescue(
                                rescue_enabled=e_tot_drift_rescue,
                                attempts_used=e_tot_drift_rescue_count,
                                max_attempts=e_tot_drift_rescue_attempts,
                            )
                        )
                        if not can_rescue:
                            run_status = "error"
                            run_error = (
                                f"NVE E_tot drift {drift:+.4f} eV "
                                f"exceeds abort threshold {e_tot_drift_threshold_eV:.4f} eV "
                                f"at step {steps} (E_tot={e_tot:.4f}, ref={e_tot_ref:.4f})"
                            )
                            c.print(Panel(
                                run_error,
                                title="[bold red]NVE energy conservation failed[/bold red]",
                                border_style="red",
                            ))
                            break

                        tricks = nve_etot_drift_rescue_tricks(e_tot_drift_rescue_count)
                        e_tot_drift_rescue_count += 1
                        c.print(Panel(
                            f"E_tot drift {drift:+.4f} eV at step {steps} "
                            f"(ref={e_tot_ref:.4f}, gate={e_tot_drift_threshold_eV:.4f}).\n"
                            f"Rescue attempt {e_tot_drift_rescue_count}/"
                            f"{e_tot_drift_rescue_attempts}: {', '.join(tricks)}",
                            title="[bold yellow]NVE E_tot drift → repair & restart[/bold yellow]",
                            border_style="yellow",
                        ))
                        pos_rescue = last_good_pos
                        if use_pbc:
                            pos_rescue = _wrap_monomers(
                                jnp.asarray(pos_rescue), _cell_jax
                            )
                        cell_for_rescue = (
                            np.asarray(jax.device_get(_cell_jax), dtype=float)
                            if use_pbc
                            else None
                        )
                        if "charmm_rescue" in tricks and overlap_charmm_rescue_fn is not None:
                            try:
                                pos_np = np.asarray(
                                    jax.device_get(pos_rescue), dtype=float
                                )
                                new_pos = overlap_charmm_rescue_fn(
                                    pos_np, cell_for_rescue
                                )
                                if new_pos is not None:
                                    pos_rescue = as_jaxmd_dtype(new_pos)
                                    charmm_overlap_rescue_count += 1
                                    c.print(
                                        "[yellow]drift rescue: CHARMM SD/ABNR applied[/yellow]"
                                    )
                            except Exception as exc:
                                c.print(
                                    f"[yellow]drift rescue: CHARMM skipped ({exc})[/yellow]"
                                )
                        if "nl_rebuild" in tricks and use_pbc and update_fn is not None:
                            try:
                                pair_i, pair_m = update_fn(
                                    _nl_update_positions(pos_rescue),
                                    box=pbc_box_nl,
                                    force_rebuild=True,
                                )
                            except TypeError:
                                pair_i, pair_m = update_fn(
                                    _nl_update_positions(pos_rescue),
                                    box=pbc_box_nl,
                                )
                            _pbc_state["pair_idx"] = pair_i
                            _pbc_state["pair_mask"] = pair_m
                            current_neighbors = (pair_i, pair_m)
                        fire_steps_this = (
                            e_tot_drift_rescue_fire_steps
                            if "fire" in tricks
                            else 0
                        )
                        if "fire" in tricks and e_tot_drift_rescue_count >= 3:
                            fire_steps_this = max(fire_steps_this, 200)
                        if fire_steps_this > 0:
                            def _drift_nl_refresh(pos):
                                if use_pbc and update_fn is not None:
                                    try:
                                        pair_i, pair_m = update_fn(
                                            _nl_update_positions(pos),
                                            box=pbc_box_nl,
                                            force_rebuild=True,
                                        )
                                    except TypeError:
                                        pair_i, pair_m = update_fn(
                                            _nl_update_positions(pos),
                                            box=pbc_box_nl,
                                        )
                                    _pbc_state["pair_idx"] = pair_i
                                    _pbc_state["pair_mask"] = pair_m

                            rescue_shift_fn = shift_min
                            if use_pbc:
                                def rescue_shift_fn(pos, dR, **kwargs):
                                    return _wrap_monomers(pos + dR, _cell_jax)

                            f0_drift = float(
                                jnp.max(
                                    jnp.linalg.norm(
                                        wrapped_force_fn(
                                            pos_rescue, neighbor=current_neighbors
                                        ),
                                        axis=-1,
                                    )
                                )
                            )
                            dt_rescue = resolve_jaxmd_fire_dt_start_ps(f0_drift)
                            pos_rescue, best_f, fire_info = run_jaxmd_fire_with_dt_backoff(
                                force_fn=wrapped_force_fn,
                                shift_fn=rescue_shift_fn,
                                positions=pos_rescue,
                                masses=Si_mass,
                                n_steps=int(fire_steps_this),
                                dt_schedule=(dt_rescue,),
                                nl_refresh_fn=(
                                    _drift_nl_refresh
                                    if (use_pbc and update_fn is not None)
                                    else None
                                ),
                                energy_fn=wrapped_energy_fn,
                            )
                            c.print(
                                Panel(
                                    f"Drift-rescue FIRE: max|F| "
                                    f"{fire_info['start_max_f']:.4f} → {best_f:.4f} eV/Å "
                                    f"({fire_steps_this} steps)",
                                    title="[bold]NVE E_tot drift rescue[/bold]",
                                    border_style="yellow",
                                )
                            )
                        if "rethermalize" in tricks:
                            state = _state_after_overlap_rescue(pos_rescue)
                        else:
                            state = state.set(position=as_jaxmd_dtype(pos_rescue))
                            state = normalize_jaxmd_state(state)
                        if use_pbc and update_fn is not None:
                            try:
                                pair_i, pair_m = update_fn(
                                    _nl_update_positions(state.position),
                                    box=pbc_box_nl,
                                    force_rebuild=True,
                                )
                            except TypeError:
                                pair_i, pair_m = update_fn(
                                    _nl_update_positions(state.position),
                                    box=pbc_box_nl,
                                )
                            _pbc_state["pair_idx"] = pair_i
                            _pbc_state["pair_mask"] = pair_m
                            current_neighbors = (pair_i, pair_m)
                        if (
                            "dt_halve" in tricks
                            and args.ensemble == "nve"
                            and not is_npt
                        ):
                            dt_old = float(dt)
                            dt_new = nve_etot_drift_halved_dt_ps(
                                dt_old,
                                scale=e_tot_drift_rescue_dt_scale,
                                min_dt_fs=e_tot_drift_rescue_min_dt_fs,
                            )
                            if dt_new < dt_old * 0.999:
                                # Preserve physical time per recording block.
                                ratio = dt_old / dt_new
                                steps_per_recording = int(
                                    max(
                                        int(steps_per_loop_call),
                                        round(int(steps_per_recording) * ratio),
                                    )
                                )
                                # Keep loop call a divisor of the new recording stride.
                                while (
                                    steps_per_recording % int(steps_per_loop_call) != 0
                                    and int(steps_per_loop_call) > 1
                                ):
                                    steps_per_loop_call = int(steps_per_loop_call) // 2
                                dt = dt_new
                                dt_fs = dt * 1000.0
                                c.print(
                                    Panel(
                                        f"Rebuilding NVE integrator: dt "
                                        f"{dt_old * 1000:.4f} → {dt_new * 1000:.4f} fs. "
                                        "First post-rescue block recompiles JAX "
                                        "(often minutes) — progress lines resume after that.",
                                        title="[bold yellow]NVE dt backoff[/bold yellow]",
                                        border_style="yellow",
                                    )
                                )
                                init_fn, apply_fn = simulate.nve(
                                    wrapped_force_fn, shift, dt_new
                                )
                                apply_fn = jit(apply_fn)
                                sim = _bind_sim(apply_fn)
                                dt = dt_new
                                dt_fs = dt * 1000.0
                                # Re-init momenta at new integrator with repaired geometry.
                                state = _state_after_overlap_rescue(state.position)
                                # Compile single-step apply_fn now (does not advance
                                # production state). The multi-step ``sim`` loop still
                                # compiles on the next recording block.
                                try:
                                    _warm = apply_fn(
                                        state, neighbor=current_neighbors
                                    )
                                    block_jax_values(
                                        _warm.position, _warm.momentum
                                    )
                                    c.print(
                                        "[yellow]drift rescue: apply_fn compiled; "
                                        "multi-step loop compiles on the next "
                                        f"record ({steps_per_recording} steps @ "
                                        f"{dt_fs:.4f} fs) — expect a pause[/yellow]"
                                    )
                                except Exception as exc:
                                    c.print(
                                        f"[yellow]drift rescue: warmup compile deferred "
                                        f"({type(exc).__name__}: {exc})[/yellow]"
                                    )
                                c.print(
                                    f"[yellow]drift rescue: MD dt "
                                    f"{dt_old * 1000:.4f} → {dt_fs:.4f} fs; "
                                    f"steps/record → {steps_per_recording} "
                                    f"(physical record stride unchanged)[/yellow]"
                                )
                            else:
                                c.print(
                                    f"[yellow]drift rescue: dt already at floor "
                                    f"{dt_fs:.4f} fs[/yellow]"
                                )
                        if not _rescued_state_energy_finite(state):
                            run_status = "error"
                            run_error = (
                                f"NVE E_tot drift rescue produced non-finite energy "
                                f"at step {steps} (attempt {e_tot_drift_rescue_count})"
                            )
                            c.print(Panel(
                                run_error,
                                title="[bold red]NVE energy conservation failed[/bold red]",
                                border_style="red",
                            ))
                            break
                        if "grace" in tricks:
                            e_tot_drift_threshold_eV = nve_etot_drift_grace_threshold_eV(
                                current_threshold_eV=e_tot_drift_threshold_eV,
                                grace_eV=e_tot_drift_rescue_grace_eV,
                                attempt_1_based=e_tot_drift_rescue_count,
                            )
                            c.print(
                                f"[yellow]drift rescue: widened E_tot gate to "
                                f"{e_tot_drift_threshold_eV:.4f} eV[/yellow]"
                            )
                        # Fresh microcanonical reference after geometry/velocity repair.
                        e_tot_ref = None
                        last_good_pos = state.position
                        force_progress_print = True
                        c.print(
                            Panel(
                                "Repair applied — continuing NVE from last-good geometry "
                                "with re-initialized velocities; E_tot reference reset. "
                                "Next progress line prints as soon as the following "
                                "recording block finishes (may be slow if dt just changed).",
                                title="[bold green]NVE E_tot drift rescue[/bold green]",
                                border_style="green",
                            )
                        )
                        continue
                    else:
                        last_good_pos = state.position
                # Cumulative MD progress (survives mid-run dt / steps_per_recording changes).
                md_steps_completed += int(steps_per_recording)
                sim_time_ps += float(steps_per_recording) * float(dt)
                steps = md_steps_completed
                time_ps = sim_time_ps
                if i % 10 == 0 or force_progress_print:
                    force_progress_print = False
                    elapsed_s = time.perf_counter() - jaxmd_loop_start
                    simulated_ns = sim_time_ps * 1e-3
                    if simulated_ns > 0 and elapsed_s > 0:
                        avg_speed_ns_per_day = simulated_ns * 86400.0 / elapsed_s
                    else:
                        avg_speed_ns_per_day = float("nan")
                    if is_npt and npt_pair_idx is not None:
                        vol = float(quantity.volume(3, box_curr))
                        box_diag = np.diagonal(np.asarray(box_curr)[:3, :3])
                        L = float(box_diag[0]) if box_diag.size > 0 else float("nan")
                        density_g_cm3 = float(np.sum(Si_mass) * 1.66053906660 / vol) if vol > 0 else float("nan")
                        BAR_PER_ATM = 1.01325
                        unit_p = float(unit["pressure"])
                        p_tgt_atm = float(npt_pressure / (unit_p * BAR_PER_ATM))
                        # Measured pressure (virial + kinetic) for diagnostics
                        try:
                            p_meas = quantity.pressure(
                                npt_energy_fn, state.position, box_curr,
                                kinetic_energy=e_kin, neighbor=(npt_pair_idx, npt_pair_mask)
                            )
                            p_meas_atm = float(p_meas / (unit_p * BAR_PER_ATM))
                        except Exception:
                            p_meas_atm = float("nan")
                        _fb_cols = (
                            f"\t{com_dist_report:8.4f}\t{e_fb_report:10.4f}"
                            if use_flat_bottom
                            else ""
                        )
                        _wall_cols = f"\t{e_wall_report:11.4f}" if report_wall else ""
                        line = (
                            f"{time_ps:10.4f}\t{steps:6d}\t{e_pot:10.4f}\t{e_tot:10.4f}\t{temp:10.2f}\t"
                            f"{L:8.2f}\t{vol:10.1f}\t{density_g_cm3:8.3f}\t{p_tgt_atm:8.2f}\t{p_meas_atm:8.2f}"
                            f"{_fb_cols}{_wall_cols}\t{avg_speed_ns_per_day:10.4f}"
                        )
                        if nbr_monitor:
                            nbr_n_valid = int(np.sum(np.asarray(jax.device_get(npt_pair_mask))))
                            nbr_capacity = npt_pair_idx.shape[0]
                            nbr_fill_ratio = nbr_n_valid / nbr_capacity if nbr_capacity > 0 else 0.0
                            line += f"\t{nbr_n_valid}\t{nbr_capacity}\t{100.0 * nbr_fill_ratio:.1f}%"
                        print(line)
                    else:
                        _fb_cols = (
                            f"\t{com_dist_report:8.4f}\t{e_fb_report:10.4f}"
                            if use_flat_bottom
                            else ""
                        )
                        _wall_cols = f"\t{e_wall_report:11.4f}" if report_wall else ""
                        print(
                            f"{time_ps:10.4f}\t{steps:6d}\t{e_pot:10.4f}\t{e_tot:10.4f}\t{temp:10.2f}"
                            f"{_fb_cols}{_wall_cols}\t{avg_speed_ns_per_day:10.4f}"
                        )

                # Record to HDF5 every record (NPT: real-space; optional monomer wrap).
                # NVE/NVT integrator coords stay unwrapped; wrap only for viz export.
                pos_for_h5 = state.position
                if is_npt:
                    box_curr = simulate.npt_box(state)
                    pos_for_h5 = space.transform(box_curr, state.position)
                    if traj_export_molecular_wrap:
                        pos_for_h5 = _wrap_monomers(pos_for_h5, box_curr)
                elif use_pbc and traj_export_molecular_wrap:
                    pos_for_h5 = _wrap_monomers(state.position, _cell_jax)
                # True extended-system invariant for NHC (not bare E_tot).
                # jax_md's nvt/npt_nose_hoover_invariant includes thermostat
                # (and barostat) degrees of freedom; bare E_tot is not conserved
                # in NVT/NpT and was previously written here by mistake.
                e_invariant = e_tot
                try:
                    if is_npt:
                        box_for_inv = simulate.npt_box(state)
                        e_invariant = float(
                            simulate.npt_nose_hoover_invariant(
                                npt_energy_fn,
                                state,
                                pressure,
                                kT,
                                neighbor=(npt_pair_idx, npt_pair_mask),
                                box=box_for_inv,
                            )
                        )
                    elif args.ensemble == "nvt":
                        e_invariant = float(
                            simulate.nvt_nose_hoover_invariant(
                                wrapped_energy_fn,
                                state,
                                kT,
                                neighbor=current_neighbors,
                            )
                        )
                except Exception as _inv_exc:
                    if not getattr(run_sim, "_invariant_warn_once", False):
                        print(
                            f"[jaxmd] NHC invariant fallback to E_tot "
                            f"({type(_inv_exc).__name__}: {_inv_exc})",
                            flush=True,
                        )
                        run_sim._invariant_warn_once = True
                    e_invariant = e_tot

                report_kw = dict(
                    potential_energy=e_pot,
                    kinetic_energy=e_kin,
                    temperature=temp,
                    invariant=e_invariant,
                    total_energy=e_tot,
                    time_ps=time_ps,
                    positions=pos_for_h5,
                    velocities=state.momentum / state.mass,
                    charges=out_dyn.mm_charges,
                )
                if is_npt:
                    box_for_density = simulate.npt_box(state)
                    vol_for_density = float(quantity.volume(3, box_for_density))
                    report_kw["density_g_cm3"] = (
                        float(np.sum(Si_mass) * 1.66053906660 / vol_for_density)
                        if vol_for_density > 0
                        else float("nan")
                    )
                if nbr_monitor and is_npt and npt_pair_idx is not None:
                    if nbr_n_valid is None:
                        nbr_n_valid = int(np.sum(np.asarray(jax.device_get(npt_pair_mask))))
                        nbr_capacity = npt_pair_idx.shape[0]
                        nbr_fill_ratio = nbr_n_valid / nbr_capacity if nbr_capacity > 0 else 0.0
                    report_kw["nbr_n_valid"] = nbr_n_valid
                    report_kw["nbr_capacity"] = nbr_capacity
                    report_kw["nbr_fill_ratio"] = nbr_fill_ratio
                if use_flat_bottom:
                    report_kw["com_dist_A"] = com_dist_report
                    report_kw["flat_bottom_E_eV"] = e_fb_report
                hdf5_reporter.report(**report_kw)

                # Stop on numerical instability (NaN, Inf, or energy blow-up to 0)
                if not np.isfinite(e_pot) or not np.isfinite(temp):
                    run_status = "error"
                    run_error = f"numerical instability at step {steps}"
                    print(f"Numerical instability at step {steps}; stopping.")
                    if len(nhc_positions) > 1:
                        nhc_positions = nhc_positions[:-1]
                        if is_npt:
                            nhc_boxes = nhc_boxes[:-1]
                    break
                if e_pot >= 0 and energy_initial < 0:
                    run_status = "error"
                    run_error = f"energy blow-up at step {steps} (E_pot={e_pot:.4f})"
                    c.print(Panel(f"Energy blow-up at step {steps} (E_pot={e_pot:.4f}); stopping.", title="[bold red]Error[/bold red]", border_style="red"))
                    if len(nhc_positions) > 1:
                        nhc_positions = nhc_positions[:-1]
                        if is_npt:
                            nhc_boxes = nhc_boxes[:-1]
                    break
        except KeyboardInterrupt:
            run_status = "interrupted"
            run_error = "KeyboardInterrupt"
            c.print(Panel("Interrupted; saving partial trajectory data.", title="[bold yellow]JAX-MD interrupted[/bold yellow]", border_style="yellow"))
        except Exception as exc:
            run_status = "error"
            run_error = f"{type(exc).__name__}: {exc}"
            c.print(Panel(f"{run_error}\nSaving partial trajectory data.", title="[bold red]JAX-MD error[/bold red]", border_style="red"))
        finally:
            try:
                hdf5_reporter.close()
            except Exception as exc:
                close_error = f"{type(exc).__name__}: {exc}"
                run_error = close_error if run_error is None else f"{run_error}; HDF5 close failed: {close_error}"
                run_status = "error"
                c.print(Panel(close_error, title="[bold red]HDF5 close failed[/bold red]", border_style="red"))
        c.print(Panel(str(hdf5_path), title="[bold green]HDF5 trajectory saved[/bold green]", border_style="green"))

        steps_completed = int(md_steps_completed) if md_steps_completed > 0 else (
            len(nhc_positions) * int(steps_per_recording)
        )
        run_sim.last_status = run_status
        run_sim.last_error = run_error
        run_sim.last_overlap_warning_count = overlap_warning_count
        run_sim.last_overlap_min_distance = overlap_min_seen
        run_sim.last_charmm_overlap_rescue_count = charmm_overlap_rescue_count
        run_sim.last_etot_drift_rescue_count = e_tot_drift_rescue_count
        if e_tot_drift_rescue_count > 0 and run_status == "complete":
            c.print(
                Panel(
                    f"NVE completed after {e_tot_drift_rescue_count} E_tot drift "
                    f"repair/restart cycle(s).",
                    title="[bold green]NVE E_tot drift rescue[/bold green]",
                    border_style="green",
                )
            )
        try:
            run_sim.last_velocities = np.asarray(
                jax.device_get(state.momentum / state.mass), dtype=float
            )
        except Exception:
            run_sim.last_velocities = None
        completion_title = "Simulation complete" if run_status == "complete" else "Partial simulation saved"
        c.print(Panel(
            f"{steps_completed} steps ({sim_time_ps:.2f} ps; dt={dt_fs:.4f} fs)",
            title=f"[bold]{completion_title}[/bold]",
            border_style="green",
        ))

        nhc_positions_out = []
        nhc_boxes_out = []  # NPT: real-space box per frame for trajectory cell
        for idx, R in enumerate(nhc_positions):
            if is_npt:
                # NPT: convert fractional to real using box at this step
                box_i = nhc_boxes[idx]
                R = space.transform(box_i, R)
                if traj_export_molecular_wrap:
                    R = _wrap_monomers(R, box_i)
                nhc_boxes_out.append(np.asarray(jax.device_get(box_i)))
            elif use_pbc:
                if pbc_map_fn is not None:
                    R = pbc_map_fn(R)
                if traj_export_molecular_wrap:
                    R = _wrap_monomers(jnp.asarray(R), _cell_jax)
            nhc_positions_out.append(np.asarray(jax.device_get(R)))
        if nhc_positions_out:
            positions_out = np.stack(nhc_positions_out)
        else:
            positions_out = np.empty((0, len(atoms), 3), dtype=np.float32)
        return steps_completed, positions_out, nhc_boxes_out if is_npt else None

    run_sim.neighbor_update_interval_steps = int(steps_per_loop_call)
    return run_sim
