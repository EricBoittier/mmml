"""Unit tests for offline restart velocity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.restart_velocity_analysis import (
    analyze_restart_velocities,
    collect_numbered_restart_paths,
    find_velocity_outliers,
)


def _minimal_restart(path: Path, *, natom: int = 4, vel: np.ndarray | None = None) -> None:
    lines = [
        "REST     0    10",
        "       1 !NTITLE followed by title",
        "* t",
        "",
        " !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NSAVL",
        f"         {natom}           0          10          10          10          10           3",
        " !X, Y, Z",
    ]
    for i in range(natom):
        lines.append(f" {i * 0.1:.15E} 0.000000000000000E+00 0.000000000000000E+00")
    if vel is not None:
        lines.append(" !VX, VY, VZ")
        for i in range(natom):
            lines.append(
                f" {vel[i, 0]:.15E} {vel[i, 1]:.15E} {vel[i, 2]:.15E}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_collect_numbered_restart_paths(tmp_path: Path) -> None:
    for i in (3, 1, 2):
        _minimal_restart(tmp_path / f"heat.{i:04d}.res")
    _minimal_restart(tmp_path / "heat.a.res")
    paths = collect_numbered_restart_paths(tmp_path, stem="heat")
    assert [p.name for p in paths] == ["heat.0001.res", "heat.0002.res", "heat.0003.res"]


def test_find_velocity_outliers_flags_hot_atom(tmp_path: Path) -> None:
    vel = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [50.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    p = tmp_path / "heat.0000.res"
    _minimal_restart(p, vel=vel)
    outliers = find_velocity_outliers(p, vel, z_threshold=3.0)
    assert outliers
    assert outliers[0].atom_index == 3


def test_analyze_restart_velocities_report(tmp_path: Path) -> None:
    vel = np.ones((2, 3), dtype=float)
    p = tmp_path / "heat.0007.res"
    _minimal_restart(p, natom=2, vel=vel)
    rep = analyze_restart_velocities(p)
    assert rep.has_velocities
    assert rep.speed_max > 0.0
