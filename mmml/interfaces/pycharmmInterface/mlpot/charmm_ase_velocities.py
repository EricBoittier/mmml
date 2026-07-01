"""Maxwell-Boltzmann velocity assignment via ASE for CHARMM dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

# amu * (Angstrom/psec)^2 -> kcal/mol (CHARMM kinetic-energy convention).
_AMU_ANG_PS2_TO_KCALMOL = 0.001036427219371
_KCALMOL_PER_K = 0.0019872041

# Hard floor for every Maxwell-Boltzmann / CHARMM ASSVEL draw (K).
MIN_VELOCITY_ASSIGNMENT_TEMP_K = 10.0


def clamp_velocity_assignment_temp_k(temperature_K: float) -> float:
    """Return ``temperature_K`` clamped to :data:`MIN_VELOCITY_ASSIGNMENT_TEMP_K`."""
    temp = float(temperature_K)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"temperature_K must be positive and finite, got {temperature_K!r}")
    return max(temp, MIN_VELOCITY_ASSIGNMENT_TEMP_K)


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    return pycharmm


def charmm_masses_amu() -> np.ndarray:
    """PSF masses (amu) for all atoms."""
    pycharmm = _import_pycharmm()
    return np.asarray(pycharmm.select.get_property("mass"), dtype=np.float64)


def charmm_velocities_akma() -> np.ndarray | None:
    """Main-set velocities as ``(N, 3)`` in CHARMM AKMA units, or ``None``."""
    from mmml.interfaces.pycharmmInterface.mlpot.run_state_checkpoint import (
        _charmm_velocities_array,
    )

    vel = _charmm_velocities_array()
    if vel is None:
        return None
    arr = np.asarray(vel, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def ase_to_charmm_akma_velocities(
    velocities_ang_fs: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Convert ASE Å/fs velocities to CHARMM AKMA components."""
    v = np.asarray(velocities_ang_fs, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(masses_amu, dtype=np.float64).reshape(-1)
    if v.shape[0] != m.shape[0]:
        raise ValueError(
            f"velocity rows ({v.shape[0]}) != mass rows ({m.shape[0]})"
        )
    scale = np.sqrt(np.maximum(m, 1.0e-12)) * 1000.0
    return v * scale[:, None]


def charmm_akma_to_ang_fs_velocities(
    velocities_akma: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    """Convert CHARMM AKMA velocities to ASE Å/fs."""
    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(masses_amu, dtype=np.float64).reshape(-1)
    scale = np.sqrt(np.maximum(m, 1.0e-12)) * 1000.0
    return v / scale[:, None]


def estimate_kinetic_temperature_k(
    velocities_akma: np.ndarray | None,
    masses_amu: np.ndarray | None = None,
    *,
    ndegf: int | None = None,
) -> float | None:
    """Instantaneous kinetic temperature (K) from CHARMM AKMA velocities."""
    if velocities_akma is None:
        return None
    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(v)):
        return None
    if masses_amu is None:
        masses_amu = charmm_masses_amu()
    m = np.asarray(masses_amu, dtype=np.float64).reshape(-1)
    if v.shape[0] != m.shape[0] or v.shape[0] == 0:
        return None
    v_ang_ps = v / (np.sqrt(np.maximum(m, 1.0e-12))[:, None] * 1000.0)
    ke_kcal = 0.5 * float(np.sum(m[:, None] * v_ang_ps * v_ang_ps)) * _AMU_ANG_PS2_TO_KCALMOL
    dof = int(ndegf) if ndegf is not None else max(1, 3 * int(v.shape[0]))
    if ke_kcal <= 0.0:
        return 0.0
    return 2.0 * ke_kcal / (float(dof) * _KCALMOL_PER_K)


def velocities_are_cold(
    velocities_akma: np.ndarray | None = None,
    *,
    masses_amu: np.ndarray | None = None,
    min_temperature_K: float = 1.0,
) -> bool:
    """True when kinetic temperature is below ``min_temperature_K``."""
    if velocities_akma is None:
        velocities_akma = charmm_velocities_akma()
    temp = estimate_kinetic_temperature_k(velocities_akma, masses_amu)
    if temp is None:
        return True
    return float(temp) < float(min_temperature_K)


def resolve_assignment_temperature_k(
    dynamics_kw: dict[str, Any] | None,
    *,
    default_K: float = 300.0,
) -> float:
    """Target Kelvin for ASE velocity assignment from dynamics keywords."""
    if not dynamics_kw:
        return clamp_velocity_assignment_temp_k(default_K)
    candidates: list[float] = []
    for key in (
        "hoover reft",
        "finalt",
        "tbath",
        "treference",
        "firstt",
        "tstruct",
    ):
        raw = dynamics_kw.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val > 0.0:
            candidates.append(val)
    if not candidates:
        return clamp_velocity_assignment_temp_k(default_K)
    return clamp_velocity_assignment_temp_k(max(candidates))


def clamp_velocity_assignment_dynamics_kw(kw: dict[str, Any]) -> None:
    """Clamp CHARMM ``FIRSTT`` / bath keys when a dyna call assigns velocities."""
    if not bool(kw.get("start")):
        return
    if int(kw.get("iasvel", 0) or 0) != 1:
        return
    for key in ("firstt", "tbath", "tstruct"):
        if key in kw:
            kw[key] = clamp_velocity_assignment_temp_k(float(kw[key]))


def sync_charmm_velocities_akma(velocities_akma: np.ndarray) -> None:
    """Write AKMA velocities into CHARMM main and COMP sets."""
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        sync_comparison_velocities_akma,
    )

    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    import pycharmm.coor as coor

    if hasattr(coor, "set_velocity"):
        coor.set_velocity(v[:, 0], v[:, 1], v[:, 2])
    sync_comparison_velocities_akma(v)


def assign_maxwell_boltzmann_velocities_via_ase(
    temperature_K: float,
    *,
    remove_net_drift: bool = True,
    remove_rotation: bool = True,
    seed: int | None = None,
    quiet: bool = False,
) -> float:
    """Assign Maxwell-Boltzmann velocities at ``temperature_K``; return estimated T."""
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution,
        Stationary,
        ZeroRotation,
    )

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ase_from_pycharmm_state

    temp = clamp_velocity_assignment_temp_k(temperature_K)

    atoms = ase_from_pycharmm_state()
    masses = charmm_masses_amu()
    if len(atoms) != len(masses):
        raise ValueError(
            f"ASE atoms ({len(atoms)}) != CHARMM masses ({len(masses)})"
        )
    atoms.set_masses(masses)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp, rng=rng)
    if remove_net_drift:
        Stationary(atoms)
    if remove_rotation:
        ZeroRotation(atoms)

    v_ase = np.asarray(atoms.get_velocities(), dtype=np.float64)
    v_akma = ase_to_charmm_akma_velocities(v_ase, masses)
    sync_charmm_velocities_akma(v_akma)

    measured = estimate_kinetic_temperature_k(v_akma, masses)
    if not quiet:
        txt = f"{measured:.2f}" if measured is not None else "?"
        print(
            f"ASE Maxwell-Boltzmann velocities at {temp:.2f} K "
            f"(measured T≈{txt} K)",
            flush=True,
        )
    return float(measured if measured is not None else temp)


def _dynamics_would_start_cold(kw: dict[str, Any]) -> bool:
    """True when CHARMM would integrate from near-zero kinetic energy."""
    iasvel = int(kw.get("iasvel", 0) or 0)
    start = bool(kw.get("start"))
    if iasvel == 0 and start:
        return True
    if iasvel == 0 and not start:
        return velocities_are_cold()
    return False


def maybe_assign_velocities_via_ase_if_cold(
    dynamics_kw: dict[str, Any] | None = None,
    *,
    temperature_K: float | None = None,
    min_temperature_K: float = 1.0,
    default_K: float = 300.0,
    quiet: bool = False,
) -> bool:
    """Assign ASE velocities when the current state is colder than ``min_temperature_K``."""
    kw = dict(dynamics_kw or {})
    if not _dynamics_would_start_cold(kw):
        return False

    target = (
        clamp_velocity_assignment_temp_k(temperature_K)
        if temperature_K is not None
        else resolve_assignment_temperature_k(kw, default_K=default_K)
    )
    assign_maxwell_boltzmann_velocities_via_ase(
        target,
        quiet=quiet,
    )

    # Continue with in-memory velocities. Use iasvel=1 so lingering START (PyCHARMM
    # omits start=False) does not read zero COMP coordinates as velocities.
    kw["iasvel"] = 1
    kw["start"] = False
    kw["iasors"] = int(kw.get("iasors", 0) or 0)
    if dynamics_kw is not None:
        dynamics_kw.clear()
        dynamics_kw.update(kw)
    return True
