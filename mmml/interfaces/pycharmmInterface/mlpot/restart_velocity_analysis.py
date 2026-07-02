"""Analyze CHARMM restart velocities (offline; no live PyCHARMM session)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
    read_restart_last_step,
    read_restart_natom,
    read_restart_velocities,
    restart_velocities_match_coordinates,
)


@dataclass(frozen=True)
class VelocityOutlier:
    restart: str
    atom_index: int
    speed_akma: float
    vx: float
    vy: float
    vz: float
    z_score: float


@dataclass(frozen=True)
class RestartVelocityReport:
    path: Path
    natom: int
    global_step: int | None
    has_velocities: bool
    coords_as_velocities: bool
    temperature_K: float | None
    speed_mean: float
    speed_std: float
    speed_max: float
    speed_p99: float
    outliers: tuple[VelocityOutlier, ...]


def collect_numbered_restart_paths(
    directory: Path,
    *,
    stem: str = "heat",
) -> list[Path]:
    """Sorted ``heat.0000.res`` siblings (excludes scratch ``heat.a.res``)."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    pattern = re.compile(rf"^{re.escape(stem)}\.(\d{{4}})\.res$")
    numbered = [
        p
        for p in d.glob(f"{stem}.*.res")
        if pattern.match(p.name)
    ]
    if numbered:
        return sorted(numbered, key=lambda p: int(pattern.match(p.name).group(1)))  # type: ignore[union-attr]
    single = d / f"{stem}.res"
    return [single] if single.is_file() else []


def _speeds_akma(vel: np.ndarray) -> np.ndarray:
    v = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    return np.linalg.norm(v, axis=1)


def _temperature_uniform_mass_k(vel: np.ndarray) -> float:
    """Kinetic T (K) with unit mass per atom (diagnostic when masses unavailable)."""
    from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
        _AMU_ANG_PS2_TO_KCALMOL,
        _KCALMOL_PER_K,
    )

    v = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
    n = v.shape[0]
    if n == 0:
        return 0.0
    v_ang_ps = v / 1000.0
    ke_kcal = 0.5 * float(np.sum(v_ang_ps * v_ang_ps)) * _AMU_ANG_PS2_TO_KCALMOL
    dof = max(1, 3 * n)
    return 2.0 * ke_kcal / (float(dof) * _KCALMOL_PER_K)


def find_velocity_outliers(
    path: Path,
    vel: np.ndarray,
    *,
    z_threshold: float = 4.0,
) -> tuple[VelocityOutlier, ...]:
    speeds = _speeds_akma(vel)
    if speeds.size == 0:
        return ()
    med = float(np.median(speeds))
    mad = float(np.median(np.abs(speeds - med)))
    scale = max(1.4826 * mad, 1.0e-8)
    z = (speeds - med) / scale
    outliers: list[VelocityOutlier] = []
    for i, (zi, sp) in enumerate(zip(z, speeds, strict=True)):
        if float(zi) < float(z_threshold):
            continue
        outliers.append(
            VelocityOutlier(
                restart=path.name,
                atom_index=int(i),
                speed_akma=float(sp),
                vx=float(vel[i, 0]),
                vy=float(vel[i, 1]),
                vz=float(vel[i, 2]),
                z_score=float(zi),
            )
        )
    outliers.sort(key=lambda o: o.z_score, reverse=True)
    return tuple(outliers)


def analyze_restart_velocities(
    path: Path,
    *,
    z_threshold: float = 4.0,
) -> RestartVelocityReport:
    """Build a velocity summary for one ``.res`` file."""
    p = Path(path)
    natom = read_restart_natom(p) or 0
    vel = read_restart_velocities(p)
    has_vel = vel is not None
    coords_bug = bool(has_vel and restart_velocities_match_coordinates(p, vel))
    if not has_vel:
        return RestartVelocityReport(
            path=p,
            natom=int(natom),
            global_step=read_restart_last_step(p),
            has_velocities=False,
            coords_as_velocities=False,
            temperature_K=None,
            speed_mean=0.0,
            speed_std=0.0,
            speed_max=0.0,
            speed_p99=0.0,
            outliers=(),
        )
    speeds = _speeds_akma(vel)
    return RestartVelocityReport(
        path=p,
        natom=int(natom),
        global_step=read_restart_last_step(p),
        has_velocities=True,
        coords_as_velocities=coords_bug,
        temperature_K=_temperature_uniform_mass_k(vel),
        speed_mean=float(np.mean(speeds)),
        speed_std=float(np.std(speeds)),
        speed_max=float(np.max(speeds)),
        speed_p99=float(np.percentile(speeds, 99)),
        outliers=find_velocity_outliers(p, vel, z_threshold=z_threshold),
    )
