"""Matched NVE stability tests for acquisition-selected potentials.

A tiny velocity-Verlet loop on the conservative student.  This is the smoke /
unit-test path and does not replace CHARMM or JAX-MD production MD.

Reported quantities:

* failures (NaN energy/force, unphysical min pair distance)
* time to first instability
* NVE total-energy drift under the *same* dt / n_steps for every method

Conservation alone does not establish physical correctness; drift is reported
alongside held-out reference accuracy, not instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

ForceFn = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]


@dataclass
class MDOutcome:
    failed: bool
    reason: str | None
    steps_completed: int
    time_to_instability: float | None
    min_pair_distance: float
    energy_drift: float
    drift_per_step: float
    max_abs_force: float
    nve_start_energy: float
    nve_end_energy: float


def min_pair_distance(R: np.ndarray, n_atoms: int) -> float:
    r = np.asarray(R, dtype=np.float64)[:n_atoms]
    if n_atoms < 2:
        return float("inf")
    dmin = np.inf
    for i in range(n_atoms):
        diff = r[i + 1 :] - r[i]
        if len(diff) == 0:
            continue
        d = np.sqrt((diff * diff).sum(axis=1))
        dmin = min(dmin, float(d.min()))
    return float(dmin)


def run_nve(
    energy_forces: ForceFn,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    masses: np.ndarray | None = None,
    dt: float = 0.5,
    n_steps: int = 40,
    min_distance: float = 0.4,
    max_force: float = 200.0,
    seed: int = 0,
    temperature: float = 300.0,
) -> MDOutcome:
    """Velocity Verlet with a kinetic seed; units are arbitrary but matched."""
    R = np.asarray(positions, dtype=np.float64).copy()
    Z = np.asarray(atomic_numbers, dtype=np.int32)
    n = int((Z > 0).sum())
    R = R[:n]
    Z = Z[:n]
    if masses is None:
        masses = np.where(Z == 1, 1.008, np.where(Z == 8, 15.999, np.where(Z == 6, 12.011, 10.0)))
    m = np.asarray(masses, dtype=np.float64).reshape(n, 1)
    rng = np.random.default_rng(int(seed))
    # Maxwell-like velocities (scale is conventional; matched across methods).
    v = rng.normal(0.0, np.sqrt(max(temperature, 1.0) / 300.0) * 0.01, size=R.shape)
    v -= v.mean(axis=0, keepdims=True)

    def _ke(vel: np.ndarray) -> float:
        return float(0.5 * np.sum(m * vel * vel))

    try:
        e, f = energy_forces(R, Z)
    except Exception as exc:  # noqa: BLE001
        return MDOutcome(
            failed=True, reason=f"initial_eval:{exc!r}", steps_completed=0,
            time_to_instability=0.0, min_pair_distance=min_pair_distance(R, n),
            energy_drift=float("nan"), drift_per_step=float("nan"),
            max_abs_force=float("nan"), nve_start_energy=float("nan"),
            nve_end_energy=float("nan"),
        )
    if not np.isfinite(e) or not np.isfinite(f).all():
        return MDOutcome(
            failed=True, reason="nonfinite_initial", steps_completed=0,
            time_to_instability=0.0, min_pair_distance=min_pair_distance(R, n),
            energy_drift=float("nan"), drift_per_step=float("nan"),
            max_abs_force=float("nan"), nve_start_energy=float("nan"),
            nve_end_energy=float("nan"),
        )
    a = f / m
    e_tot0 = e + _ke(v)
    e_last = e_tot0
    dmin = min_pair_distance(R, n)
    fmax = float(np.max(np.abs(f)))
    for step in range(int(n_steps)):
        v = v + 0.5 * dt * a
        R = R + dt * v
        dmin = min(dmin, min_pair_distance(R, n))
        if dmin < min_distance:
            return MDOutcome(
                failed=True, reason="unphysical_min_distance",
                steps_completed=step + 1, time_to_instability=dt * (step + 1),
                min_pair_distance=dmin, energy_drift=float("nan"),
                drift_per_step=float("nan"), max_abs_force=fmax,
                nve_start_energy=e_tot0, nve_end_energy=e_last,
            )
        try:
            e, f = energy_forces(R, Z)
        except Exception as exc:  # noqa: BLE001
            return MDOutcome(
                failed=True, reason=f"eval:{exc!r}", steps_completed=step + 1,
                time_to_instability=dt * (step + 1), min_pair_distance=dmin,
                energy_drift=float("nan"), drift_per_step=float("nan"),
                max_abs_force=fmax, nve_start_energy=e_tot0, nve_end_energy=e_last,
            )
        if not np.isfinite(e) or not np.isfinite(f).all():
            return MDOutcome(
                failed=True, reason="nonfinite", steps_completed=step + 1,
                time_to_instability=dt * (step + 1), min_pair_distance=dmin,
                energy_drift=float("nan"), drift_per_step=float("nan"),
                max_abs_force=fmax, nve_start_energy=e_tot0, nve_end_energy=e_last,
            )
        fmax = max(fmax, float(np.max(np.abs(f))))
        if fmax > max_force:
            return MDOutcome(
                failed=True, reason="force_explosion", steps_completed=step + 1,
                time_to_instability=dt * (step + 1), min_pair_distance=dmin,
                energy_drift=float("nan"), drift_per_step=float("nan"),
                max_abs_force=fmax, nve_start_energy=e_tot0, nve_end_energy=e_last,
            )
        a = f / m
        v = v + 0.5 * dt * a
        e_last = e + _ke(v)
    drift = e_last - e_tot0
    return MDOutcome(
        failed=False, reason=None, steps_completed=int(n_steps),
        time_to_instability=None, min_pair_distance=dmin, energy_drift=float(drift),
        drift_per_step=float(drift / max(n_steps, 1)), max_abs_force=fmax,
        nve_start_energy=float(e_tot0), nve_end_energy=float(e_last),
    )


def outcome_to_dict(out: MDOutcome) -> dict[str, Any]:
    return {
        "failed": out.failed,
        "reason": out.reason,
        "steps_completed": out.steps_completed,
        "time_to_instability": out.time_to_instability,
        "min_pair_distance": out.min_pair_distance,
        "energy_drift": out.energy_drift,
        "drift_per_step": out.drift_per_step,
        "max_abs_force": out.max_abs_force,
        "nve_start_energy": out.nve_start_energy,
        "nve_end_energy": out.nve_end_energy,
    }
