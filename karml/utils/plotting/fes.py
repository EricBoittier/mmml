"""Free-energy profiles and surfaces from arbitrary trajectory coordinates."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Coordinate = Callable[[Atoms], float]

_GAS_CONSTANTS = {
    "kcal/mol": 0.00198720425864083,
    "kJ/mol": 0.00831446261815324,
    "eV": 8.617333262145e-5,
}


@dataclass(frozen=True)
class FreeEnergySurface:
    """Histogram-derived free energy on one or two coordinate axes."""

    coordinates: tuple[np.ndarray, ...]
    free_energy: np.ndarray
    probability: np.ndarray
    edges: tuple[np.ndarray, ...]
    temperature_k: float
    energy_unit: str


def evaluate_coordinates(
    trajectory: Iterable[Atoms], coordinates: Sequence[Coordinate]
) -> np.ndarray:
    """Evaluate one or two scalar coordinate functions for every frame."""
    if len(coordinates) not in (1, 2):
        raise ValueError("exactly one or two coordinates are required")
    values = np.asarray(
        [[coordinate(atoms) for coordinate in coordinates] for atoms in trajectory],
        dtype=float,
    )
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("trajectory is empty")
    return values


def calculate_fes(
    samples: np.ndarray,
    *,
    temperature_k: float = 300.0,
    bins: int | Sequence[int] = 72,
    ranges: Sequence[tuple[float, float]] | None = None,
    weights: np.ndarray | None = None,
    energy_unit: str = "kcal/mol",
    minimum_probability: float | None = None,
) -> FreeEnergySurface:
    """Calculate ``F(q) = -R T ln P(q)`` for one- or two-dimensional samples."""
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 2):
        raise ValueError("samples must have shape (n,), (n, 1), or (n, 2)")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if energy_unit not in _GAS_CONSTANTS:
        raise ValueError(f"energy_unit must be one of {tuple(_GAS_CONSTANTS)}")

    sample_weights = None if weights is None else np.asarray(weights, dtype=float)
    valid = np.all(np.isfinite(values), axis=1)
    if sample_weights is not None:
        if sample_weights.shape != (len(values),):
            raise ValueError("weights must have one value per sample")
        valid &= np.isfinite(sample_weights) & (sample_weights >= 0)
        sample_weights = sample_weights[valid]
    values = values[valid]
    if not len(values):
        raise ValueError("no finite coordinate samples are available")

    histogram, edges = np.histogramdd(
        values, bins=bins, range=ranges, weights=sample_weights, density=False
    )
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("histogram has zero total weight")
    probability = histogram / total
    positive = probability > 0
    floor = (
        float(minimum_probability)
        if minimum_probability is not None
        else max(np.finfo(float).tiny, float(probability[positive].min()) * 0.5)
    )
    free_energy = -_GAS_CONSTANTS[energy_unit] * temperature_k * np.log(
        np.maximum(probability, floor)
    )
    free_energy -= np.min(free_energy[positive])
    centers = tuple(0.5 * (edge[:-1] + edge[1:]) for edge in edges)
    return FreeEnergySurface(
        coordinates=centers,
        free_energy=free_energy,
        probability=probability,
        edges=tuple(edges),
        temperature_k=temperature_k,
        energy_unit=energy_unit,
    )


def fes_from_trajectory(
    trajectory: Iterable[Atoms], coordinates: Sequence[Coordinate], **kwargs
) -> FreeEnergySurface:
    """Evaluate arbitrary coordinates and calculate their free-energy surface."""
    return calculate_fes(evaluate_coordinates(trajectory, coordinates), **kwargs)


def plot_fes(
    surface: FreeEnergySurface,
    *,
    labels: Sequence[str] | None = None,
    max_free_energy: float | None = None,
    cmap: str = "magma",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a one-dimensional profile or two-dimensional surface."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 5.5))
    else:
        figure = ax.figure
    names = tuple(labels or ("Coordinate 1", "Coordinate 2"))
    energy = np.where(surface.probability > 0, surface.free_energy, np.nan)
    if max_free_energy is not None:
        energy = np.minimum(energy, max_free_energy)
    if len(surface.coordinates) == 1:
        ax.plot(surface.coordinates[0], energy)
        ax.set_ylabel(f"Free energy ({surface.energy_unit})")
    else:
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#eeeeee")
        image = ax.pcolormesh(
            surface.edges[0], surface.edges[1], energy.T, shading="flat", cmap=color_map
        )
        figure.colorbar(image, ax=ax, label=f"Free energy ({surface.energy_unit})")
        ax.set_ylabel(names[1])
    ax.set_xlabel(names[0])
    ax.set_title(f"Free-energy surface at {surface.temperature_k:g} K")
    figure.tight_layout()
    return figure, ax
