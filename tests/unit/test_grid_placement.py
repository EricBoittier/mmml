"""Unit tests for grid-based composition placement helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_resolve_system_builder_defaults_and_overrides():
    from mmml.interfaces.pycharmmInterface.grid_placement import resolve_system_builder

    assert resolve_system_builder(composition=None) == "gas"
    assert resolve_system_builder(composition="DCM:8") == "liquid"
    assert resolve_system_builder(composition="DCM:8", pyxtal=True) == "crystal"
    assert resolve_system_builder(builder="gas", composition="DCM:8") == "gas"

    with pytest.raises(ValueError, match="Invalid builder"):
        resolve_system_builder(builder="packmol")


def test_grid_centers_cube_stay_inside_centered_cube():
    from mmml.interfaces.pycharmmInterface.grid_placement import grid_centers_cube

    centers = grid_centers_cube(
        8,
        center=(1.0, 2.0, 3.0),
        side=4.0,
        seed=123,
    )

    assert centers.shape == (8, 3)
    assert np.all(centers[:, 0] >= -1.0)
    assert np.all(centers[:, 0] <= 3.0)
    assert np.all(centers[:, 1] >= 0.0)
    assert np.all(centers[:, 1] <= 4.0)
    assert np.all(centers[:, 2] >= 1.0)
    assert np.all(centers[:, 2] <= 5.0)


def test_grid_centers_sphere_stay_inside_radius():
    from mmml.interfaces.pycharmmInterface.grid_placement import grid_centers_sphere

    center = np.array([1.0, -2.0, 0.5], dtype=float)
    centers = grid_centers_sphere(12, center=tuple(center), radius=5.0, seed=7)

    assert centers.shape == (12, 3)
    assert np.max(np.linalg.norm(centers - center, axis=1)) <= 5.0 + 1.0e-12

