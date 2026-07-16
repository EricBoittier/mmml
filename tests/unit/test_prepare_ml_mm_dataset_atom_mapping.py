"""Regression tests for CGenFF template-to-geometry atom correspondence."""

from __future__ import annotations

import numpy as np

from scripts.prepare_ml_mm_dataset import match_cgenff_template_fast


def _water_ohh() -> tuple[np.ndarray, np.ndarray]:
    z = np.array([8, 1, 1], dtype=np.int32)
    r = np.array(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9270, 0.0]],
        dtype=np.float64,
    )
    return z, r


def test_tip3_parameters_follow_permuted_geometry_atom_order() -> None:
    z, r = _water_ohh()
    permutation = np.array([1, 2, 0])  # observed H,H,O instead of template O,H,H

    residue, _, charges = match_cgenff_template_fast(z[permutation], r[permutation])

    assert residue == "TIP3"
    np.testing.assert_allclose(charges, [0.417, 0.417, -0.834])


def test_tip3_parameters_preserve_template_order_geometry() -> None:
    z, r = _water_ohh()

    residue, _, charges = match_cgenff_template_fast(z, r)

    assert residue == "TIP3"
    np.testing.assert_allclose(charges, [-0.834, 0.417, 0.417])
