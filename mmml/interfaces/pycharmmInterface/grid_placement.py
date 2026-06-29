"""Grid-based initial placement for gas/liquid composition builders."""

from __future__ import annotations

from typing import Literal

import numpy as np

GridPlacement = Literal["cube", "sphere"]
SystemBuilder = Literal["gas", "liquid", "crystal"]


def resolve_system_builder(
    *,
    builder: str | None = None,
    composition: str | None = None,
    pyxtal: bool | None = None,
) -> SystemBuilder:
    """Resolve the high-level starting-coordinate builder."""
    if builder is not None:
        name = str(builder).strip().lower()
        if name in ("gas", "liquid", "crystal"):
            return name  # type: ignore[return-value]
        raise ValueError(
            f"Invalid builder {builder!r}; expected 'gas', 'liquid', or 'crystal'."
        )
    if pyxtal is True:
        return "crystal"
    if composition is not None:
        return "liquid"
    return "gas"


def resolve_grid_placement_mode(
    *,
    placement: str | None = None,
    sphere: bool | None = None,
) -> GridPlacement:
    """Return the geometric placement constraint for grid builders."""
    if placement is not None:
        mode = str(placement).strip().lower()
        if mode in ("cube", "sphere"):
            return mode  # type: ignore[return-value]
        raise ValueError(f"Invalid placement {placement!r}; expected 'cube' or 'sphere'.")
    if sphere is True:
        return "sphere"
    return "cube"


def grid_centers_cube(
    n_centers: int,
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    side: float | None = None,
    spacing: float = 4.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return up to ``n_centers`` COM centers on a 3D cubic grid."""
    n = int(n_centers)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if side is not None and float(side) > 0.0:
        n_axis = int(np.ceil(n ** (1.0 / 3.0)))
        step = float(side) / float(n_axis)
        origin = np.asarray(center, dtype=float) - 0.5 * float(side)
        values = origin + (np.arange(n_axis, dtype=float) + 0.5) * step
    else:
        n_axis = int(np.ceil(n ** (1.0 / 3.0)))
        step = max(float(spacing), 1.0e-6)
        axis_values = []
        for axis_center in np.asarray(center, dtype=float):
            axis_values.append(
                (np.arange(n_axis, dtype=float) - 0.5 * (n_axis - 1)) * step
                + float(axis_center)
            )
        mesh = np.meshgrid(*axis_values, indexing="ij")
        pts = np.column_stack([m.reshape(-1) for m in mesh])
        return _select_centers(pts, n, seed=seed)
    mesh = np.meshgrid(values, values, values, indexing="ij")
    pts = np.column_stack([m.reshape(-1) for m in mesh])
    return _select_centers(pts, n, seed=seed)


def grid_centers_sphere(
    n_centers: int,
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float,
    seed: int | None = None,
) -> np.ndarray:
    """Return up to ``n_centers`` COM centers inside a sphere."""
    n = int(n_centers)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    r = float(radius)
    if r <= 0.0:
        raise ValueError(f"sphere radius must be positive, got {radius}")
    center_arr = np.asarray(center, dtype=float)
    n_axis = max(1, int(np.ceil((6.0 * n / np.pi) ** (1.0 / 3.0))))
    while True:
        step = (2.0 * r) / float(n_axis)
        values = -r + (np.arange(n_axis, dtype=float) + 0.5) * step
        mesh = np.meshgrid(values, values, values, indexing="ij")
        rel = np.column_stack([m.reshape(-1) for m in mesh])
        inside = rel[np.linalg.norm(rel, axis=1) <= r + 1.0e-9]
        if int(inside.shape[0]) >= n:
            return _select_centers(inside + center_arr, n, seed=seed)
        n_axis += 1


def _select_centers(points: np.ndarray, n_centers: int, *, seed: int | None) -> np.ndarray:
    if int(points.shape[0]) < int(n_centers):
        raise ValueError(
            f"Grid generated {points.shape[0]} centers, fewer than requested {n_centers}."
        )
    order = np.arange(int(points.shape[0]))
    if seed is not None:
        np.random.default_rng(int(seed)).shuffle(order)
    selected = points[order[: int(n_centers)]]
    # Stable output order keeps residue/chunk placement deterministic after seeded selection.
    return selected[np.lexsort((selected[:, 2], selected[:, 1], selected[:, 0]))]

