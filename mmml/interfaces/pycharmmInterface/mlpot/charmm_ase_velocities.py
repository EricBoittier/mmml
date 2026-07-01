"""Maxwell-Boltzmann and Bussi velocity control via ASE for CHARMM dynamics."""

from __future__ import annotations

import math
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

# amu * (Angstrom/psec)^2 -> kcal/mol (CHARMM kinetic-energy convention).
_AMU_ANG_PS2_TO_KCALMOL = 0.001036427219371
_KCALMOL_PER_K = 0.0019872041

# Hard floor for every Maxwell-Boltzmann / CHARMM ASSVEL draw (K).
MIN_VELOCITY_ASSIGNMENT_TEMP_K = 10.0
MAX_BUSSI_RESCALE_ALPHA = 3.0

_last_synced_velocities_akma: np.ndarray | None = None


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
    """Main-set velocities as ``(N, 3)`` in CHARMM AKMA units, or ``None``.

    Uses ``coor.get_velocity`` only. COMP is not a reliable velocity source:
    with ``iasvel=1`` dynamics it often holds comparison coordinates (positions).
    """
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


def last_synced_velocities_akma_raw() -> np.ndarray | None:
    """Last AKMA array from :func:`sync_charmm_velocities_akma` (no temperature gate)."""
    global _last_synced_velocities_akma
    if _last_synced_velocities_akma is None:
        return None
    vel = np.asarray(_last_synced_velocities_akma, dtype=np.float64).reshape(-1, 3)
    if vel.size == 0 or not np.all(np.isfinite(vel)):
        return None
    return vel


def charmm_synced_velocities_akma() -> np.ndarray | None:
    """Return the last AKMA velocities written by :func:`sync_charmm_velocities_akma`."""
    vel = last_synced_velocities_akma_raw()
    if vel is None or velocities_are_cold(vel):
        return None
    return vel


def resolve_restart_velocities_read_paths(
    path: Path | str | None,
) -> list[Path]:
    """Candidate restart paths that may hold ``!VELOCITIES`` (staging + overlap slots)."""
    if path is None:
        return []
    p = Path(path).expanduser()
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path | None) -> None:
        if candidate is None:
            return
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    _add(p)
    if p.suffix.lower() == ".res":
        try:
            from mmml.interfaces.pycharmmInterface.mlpot.artifact_paths import (
                overlap_restart_slot_paths,
            )

            slot_a, slot_b = overlap_restart_slot_paths(p)
            _add(slot_a)
            _add(slot_b)
        except Exception:
            pass
        try:
            from mmml.interfaces.pycharmmInterface.charmm_paths import charmm_io_staging_root

            tag = sha256(str(p.resolve()).encode()).hexdigest()[:16]
            staging = charmm_io_staging_root() / tag / p.name.lower()
            _add(staging)
        except Exception:
            pass
    return candidates


def charmm_velocities_akma_for_thermostat() -> np.ndarray | None:
    """Best-effort AKMA velocities for Bussi / mirror (cache, main, COMP)."""
    synced = charmm_synced_velocities_akma()
    if synced is not None:
        return synced
    vel = charmm_velocities_akma()
    if vel is not None and not velocities_are_cold(vel):
        return vel
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
            comparison_matches_main_positions,
            comparison_velocities_akma,
        )

        comp = comparison_velocities_akma()
        if (
            comp is not None
            and not comparison_matches_main_positions()
            and not velocities_are_cold(comp)
        ):
            return comp
    except Exception:
        pass
    return vel


def capture_charmm_velocities_for_bussi(
    *,
    restart_path: Path | str | None = None,
) -> np.ndarray | None:
    """Load AKMA velocities into main/COMP/cache for Bussi (cache, restart, then memory)."""
    raw = last_synced_velocities_akma_raw()
    if raw is not None and not velocities_are_cold(raw):
        return raw

    vel = _read_restart_velocities_akma(restart_path)
    if vel is not None:
        sync_charmm_velocities_akma(vel)
        return vel

    vel = charmm_velocities_akma_for_thermostat()
    if vel is not None and not velocities_are_cold(vel):
        sync_charmm_velocities_akma(vel)
        return vel
    try:
        from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
            comparison_matches_main_positions,
            comparison_velocities_akma,
        )

        comp = comparison_velocities_akma()
        if (
            comp is not None
            and not comparison_matches_main_positions()
            and not velocities_are_cold(comp)
        ):
            sync_charmm_velocities_akma(comp)
            return comp
    except Exception:
        pass
    return None


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
    if velocities_akma is None:
        return True
    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return True
    if float(np.max(np.abs(v))) < 1.0e-8:
        return True
    temp = estimate_kinetic_temperature_k(velocities_akma, masses_amu)
    if temp is None:
        # Non-zero AKMA components but CHARMM masses unavailable (unit tests / early import).
        return False
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


def _pycharmm_coor_module():
    import pycharmm.coor as coor

    return coor


def sync_charmm_velocities_akma(velocities_akma: np.ndarray) -> None:
    """Write AKMA velocities into CHARMM main and COMP sets."""
    global _last_synced_velocities_akma
    from mmml.interfaces.pycharmmInterface.mlpot.comp_velocities import (
        sync_comparison_velocities_akma,
    )

    v = np.asarray(velocities_akma, dtype=np.float64).reshape(-1, 3)
    _last_synced_velocities_akma = v.copy()
    coor = _pycharmm_coor_module()

    if hasattr(coor, "set_velocity"):
        coor.set_velocity(v[:, 0], v[:, 1], v[:, 2])
    sync_comparison_velocities_akma(v)


def _maxwell_boltzmann_akma_numpy(
    masses_amu: np.ndarray,
    temperature_K: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw Maxwell-Boltzmann AKMA velocities without ASE/CHARMM readback."""
    m = np.asarray(masses_amu, dtype=np.float64).reshape(-1)
    if m.size == 0:
        raise ValueError("masses_amu must be non-empty")
    temp = clamp_velocity_assignment_temp_k(temperature_K)
    gen = rng if rng is not None else np.random.default_rng()
    kT_kcal = _KCALMOL_PER_K * temp
    std_ang_fs = np.sqrt(
        kT_kcal / (_AMU_ANG_PS2_TO_KCALMOL * np.maximum(m, 1.0e-12))
    )
    v_ang_fs = gen.normal(0.0, std_ang_fs[:, None], size=(m.shape[0], 3))
    v_ang_fs -= v_ang_fs.mean(axis=0, keepdims=True)
    return ase_to_charmm_akma_velocities(v_ang_fs, m)


def assign_maxwell_boltzmann_velocities_via_ase(
    temperature_K: float,
    *,
    remove_net_drift: bool = True,
    remove_rotation: bool = True,
    seed: int | None = None,
    quiet: bool = False,
) -> tuple[float, np.ndarray]:
    """Assign Maxwell-Boltzmann velocities at ``temperature_K``.

    Returns ``(estimated_T, v_akma)``.
    """
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
    return float(measured if measured is not None else temp), v_akma


def assign_bussi_fallback_velocities(
    temperature_K: float,
    *,
    seed: int | None = None,
    quiet: bool = False,
) -> tuple[float, np.ndarray]:
    """Assign warm AKMA velocities for Bussi (ASE, then in-memory numpy fallback)."""
    temp = clamp_velocity_assignment_temp_k(temperature_K)
    try:
        return assign_maxwell_boltzmann_velocities_via_ase(
            temp,
            seed=seed,
            quiet=quiet,
        )
    except Exception as exc:
        if not quiet:
            print(
                f"ASE Maxwell-Boltzmann assign failed ({exc}); "
                f"drawing in-memory Maxwell-Boltzmann at {temp:.2f} K",
                flush=True,
            )
        masses = charmm_masses_amu()
        rng = np.random.default_rng(seed)
        v_akma = _maxwell_boltzmann_akma_numpy(masses, temp, rng=rng)
        sync_charmm_velocities_akma(v_akma)
        measured = estimate_kinetic_temperature_k(v_akma, masses)
        if not quiet:
            txt = f"{measured:.2f}" if measured is not None else "?"
            print(
                f"In-memory Maxwell-Boltzmann velocities at {temp:.2f} K "
                f"(measured T≈{txt} K)",
                flush=True,
            )
        return float(measured if measured is not None else temp), v_akma


def _read_restart_velocities_akma(path: Path | str | None) -> np.ndarray | None:
    """Load finite ``!VELOCITIES`` from restart candidates (staging + overlap slots)."""
    if path is None:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        read_restart_velocities,
    )

    for candidate in resolve_restart_velocities_read_paths(path):
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            continue
        vel = read_restart_velocities(candidate)
        if vel is None or not np.all(np.isfinite(vel)):
            continue
        if float(np.max(np.abs(vel))) < 1.0e-8:
            continue
        return np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    return None


def ensure_bussi_velocities_after_overlap_recovery(
    restart_path: Path | str | None,
    *,
    temperature_K: float,
    quiet: bool = True,
) -> bool:
    """Rehydrate AKMA velocities after overlap rescue / CGENFF reregister."""
    if capture_charmm_velocities_for_bussi(restart_path=restart_path) is not None:
        return True
    vel = _read_restart_velocities_akma(restart_path)
    if vel is not None:
        sync_charmm_velocities_akma(vel)
        return True
    assign_bussi_fallback_velocities(temperature_K, quiet=quiet)
    return last_synced_velocities_akma_raw() is not None


def _resolve_bussi_rescale_velocities(
    *,
    restart_path: Path | str | None,
    temperature_K: float,
    quiet: bool,
) -> np.ndarray:
    """Best-effort AKMA velocities for one Bussi rescale (never returns ``None``)."""
    temp = clamp_velocity_assignment_temp_k(temperature_K)
    vel = capture_charmm_velocities_for_bussi(restart_path=restart_path)
    if vel is not None:
        return np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    vel = _read_restart_velocities_akma(restart_path)
    if vel is not None:
        sync_charmm_velocities_akma(vel)
        return vel
    raw = last_synced_velocities_akma_raw()
    if raw is not None:
        return raw
    if not quiet:
        print(
            f"ASE Bussi rescale: no readable velocities; "
            f"assigning Maxwell-Boltzmann at {temp:.2f} K",
            flush=True,
        )
    _, vel = assign_bussi_fallback_velocities(temp, quiet=quiet)
    return np.asarray(vel, dtype=np.float64).reshape(-1, 3)


def _dynamics_would_start_cold(kw: dict[str, Any]) -> bool:
    """True when CHARMM would integrate from near-zero kinetic energy."""
    iasvel = int(kw.get("iasvel", 0) or 0)
    start = bool(kw.get("start"))
    if iasvel == 0 and start:
        return True
    if iasvel == 0 and not start:
        return velocities_are_cold()
    return False


def estimate_kinetic_energy_kcalmol(
    velocities_akma: np.ndarray | None,
    masses_amu: np.ndarray | None = None,
) -> float | None:
    """Kinetic energy (kcal/mol) from CHARMM AKMA velocities."""
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
    return float(
        0.5 * np.sum(m[:, None] * v_ang_ps * v_ang_ps) * _AMU_ANG_PS2_TO_KCALMOL
    )


def target_kinetic_energy_kcalmol(temperature_K: float, ndof: int) -> float:
    """Canonical target kinetic energy (kcal/mol) at ``temperature_K``."""
    dof = max(1, int(ndof))
    return 0.5 * dof * _KCALMOL_PER_K * float(temperature_K)


def calculate_bussi_rescale_alpha(
    kinetic_energy: float,
    *,
    target_kinetic_energy: float,
    ndof: int,
    coupling_time_ps: float,
    elapsed_time_ps: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Stochastic velocity scaling factor (Bussi et al., JCP 126, 014101).

    ``coupling_time_ps`` is the Bussi ``taut``; ``elapsed_time_ps`` is the time
    since the last rescale (typically ``rescale_interval * timestep``).
    """
    ke = float(kinetic_energy)
    target_ke = float(target_kinetic_energy)
    dof = max(1, int(ndof))
    if ke <= 0.0 or not np.isfinite(ke):
        return 1.0
    taut = max(1.0e-12, float(coupling_time_ps))
    dt = max(1.0e-12, float(elapsed_time_ps))
    exp_term = math.exp(-dt / taut)
    energy_scaling_term = (1.0 - exp_term) * target_ke / ke / dof
    gen = rng if rng is not None else np.random.default_rng()
    normal_noise = float(gen.standard_normal())
    sum_of_noises = float(2.0 * gen.standard_gamma(0.5 * (dof - 1)))
    return math.sqrt(
        exp_term
        + energy_scaling_term * (sum_of_noises + normal_noise**2)
        + 2.0 * normal_noise * math.sqrt(max(0.0, exp_term * energy_scaling_term))
    )


def resolve_bussi_degrees_of_freedom(ndegf: int | None = None) -> int:
    """Degrees of freedom for Bussi rescaling (CHARMM ``NDEGF`` when known)."""
    if ndegf is not None:
        val = int(ndegf)
        if val > 0:
            return val
    return max(1, 3 * int(charmm_masses_amu().shape[0]))


def apply_bussi_velocity_rescale(
    temperature_K: float,
    *,
    timestep_ps: float,
    rescale_interval_steps: int = 1,
    taut_ps: float | None = None,
    ndegf: int | None = None,
    quiet: bool = False,
    rng: np.random.Generator | None = None,
    restart_path: Path | str | None = None,
) -> tuple[float, float]:
    """Rescale CHARMM velocities toward ``temperature_K`` using Bussi dynamics.

    Returns ``(measured_T_after, alpha)``.
    """
    temp = clamp_velocity_assignment_temp_k(temperature_K)
    interval = max(1, int(rescale_interval_steps))
    dt_ps = max(1.0e-12, float(timestep_ps))
    elapsed_ps = interval * dt_ps
    taut = float(taut_ps) if taut_ps is not None else elapsed_ps
    masses = charmm_masses_amu()
    v_akma = _resolve_bussi_rescale_velocities(
        restart_path=restart_path,
        temperature_K=temp,
        quiet=quiet,
    )
    dof = resolve_bussi_degrees_of_freedom(ndegf)
    ke = estimate_kinetic_energy_kcalmol(v_akma, masses)
    if ke is None or ke <= 0.0:
        _, v_akma = assign_bussi_fallback_velocities(temp, quiet=quiet)
        ke = estimate_kinetic_energy_kcalmol(v_akma, masses)
        if ke is None or ke <= 0.0:
            ke = target_kinetic_energy_kcalmol(temp, dof)
    target_ke = target_kinetic_energy_kcalmol(temp, dof)
    alpha = calculate_bussi_rescale_alpha(
        ke,
        target_kinetic_energy=target_ke,
        ndof=dof,
        coupling_time_ps=taut,
        elapsed_time_ps=elapsed_ps,
        rng=rng,
    )
    t_before = estimate_kinetic_temperature_k(v_akma, masses, ndegf=dof)
    if (
        not np.isfinite(alpha)
        or alpha <= 0.0
        or alpha > MAX_BUSSI_RESCALE_ALPHA
        or alpha < 1.0 / MAX_BUSSI_RESCALE_ALPHA
    ):
        if t_before is not None and float(t_before) >= MIN_VELOCITY_ASSIGNMENT_TEMP_K:
            alpha = float(np.sqrt(max(temp, MIN_VELOCITY_ASSIGNMENT_TEMP_K) / float(t_before)))
        alpha = float(
            np.clip(alpha if np.isfinite(alpha) and alpha > 0.0 else 1.0, 1.0 / MAX_BUSSI_RESCALE_ALPHA, MAX_BUSSI_RESCALE_ALPHA)
        )
        if not quiet:
            print(
                f"ASE Bussi rescale: capped α={alpha:.4f} "
                f"(T_before≈{t_before:.2f} K → target {temp:.2f} K)",
                flush=True,
            )
    v_akma = np.asarray(v_akma, dtype=np.float64) * float(alpha)
    sync_charmm_velocities_akma(v_akma)
    measured = estimate_kinetic_temperature_k(v_akma, masses, ndegf=dof)
    if not quiet:
        txt_meas = f"{measured:.2f}" if measured is not None else "?"
        print(
            f"ASE Bussi rescale toward {temp:.2f} K "
            f"(T_after≈{txt_meas} K, α={alpha:.4f})",
            flush=True,
        )
    return (
        float(measured if measured is not None else temp),
        float(alpha),
    )


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
