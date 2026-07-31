"""Unit tests for DihedralCV + periodic umbrella bias."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.md.restraints import DihedralCV, cv_from_spec, periodic_delta_deg
from mmml.umbrella.energy import (
    numpy_bias_matrix_cv,
    packed_bias_energies_cv,
    packed_bias_forces_cv,
)


def _planar_frame() -> np.ndarray:
    """Four atoms with a known dihedral near 0° (trans-ish)."""
    return np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_cv_from_spec_dihedral():
    cv = cv_from_spec({"kind": "dihedral", "atoms": [14, 16, 18, 24]})
    assert isinstance(cv, DihedralCV)
    assert cv.atoms == (14, 16, 18, 24)
    assert cv_from_spec((0, 1, 2, 3)).atoms == (0, 1, 2, 3)


def test_periodic_delta_wraps():
    d = float(periodic_delta_deg(jnp.asarray(170.0), jnp.asarray(-170.0)))
    assert abs(d - (-20.0)) < 1e-6 or abs(d - 20.0) < 1e-6


def test_dihedral_value_and_bias_forces():
    r = _planar_frame()
    cv = DihedralCV(atoms=(0, 1, 2, 3))
    phi = cv.value_numpy(r)
    assert np.isfinite(phi)

    # Pack two windows with the same geometry
    packed = np.vstack([r, r])
    targets = (float(phi), float(phi) + 30.0)
    ks = (0.05, 0.05)  # eV/deg^2
    e = np.asarray(packed_bias_energies_cv(jnp.asarray(packed), 4, cv, targets, ks))
    assert e.shape == (2,)
    assert e[0] < 1e-8
    assert e[1] > 0.0

    f = np.asarray(packed_bias_forces_cv(jnp.asarray(packed), 4, cv, targets, ks))
    assert f.shape == (8, 3)
    assert np.all(np.isfinite(f))
    # On-target window ≈ zero bias force
    assert float(np.max(np.abs(f[:4]))) < 1e-5

    w = numpy_bias_matrix_cv(r, cv, targets, ks)
    assert w.shape == (2,)
    assert w[0] < 1e-8
