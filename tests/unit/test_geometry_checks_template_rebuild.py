"""Peer-template rebuild for high-force monomers (jax-md FIRE worst case)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.utils.geometry_checks import (
    monomer_max_force_magnitudes,
    rebuild_high_force_monomers_from_peers,
)


def _water(com, angle_deg: float = 104.5, oh: float = 0.957):
    """Simple planar water at ``com`` with H–O–H = ``angle_deg``."""
    half = np.radians(angle_deg) / 2.0
    o = np.asarray(com, dtype=float)
    h1 = o + oh * np.array([np.sin(half), np.cos(half), 0.0])
    h2 = o + oh * np.array([-np.sin(half), np.cos(half), 0.0])
    return np.vstack([o, h1, h2])


def test_rebuild_high_force_monomers_from_peers_restores_healthy_angle():
    healthy = _water([0.0, 0.0, 0.0], angle_deg=104.5)
    crushed = _water([5.0, 0.0, 0.0], angle_deg=40.0)
    ok2 = _water([10.0, 0.0, 0.0], angle_deg=104.5)
    pos = np.vstack([healthy, crushed, ok2])
    offsets = np.array([0, 3, 6, 9], dtype=int)
    forces = np.zeros_like(pos)
    forces[3:6] = 10.0  # crushed monomer carries the force

    new_pos, victims, donor = rebuild_high_force_monomers_from_peers(
        pos,
        forces,
        offsets,
        force_percentile=50.0,
        max_rebuild=2,
        min_force_eVA=1.0,
    )
    assert victims == [1]
    assert donor in (0, 2)

    def hoh(chunk):
        o, h1, h2 = chunk
        v1, v2 = h1 - o, h2 - o
        c = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        return float(np.degrees(np.arccos(c)))

    assert hoh(new_pos[3:6]) == pytest.approx(104.5, abs=1.0)
    # COM of rebuilt monomer preserved.
    assert np.allclose(new_pos[3:6].mean(axis=0), crushed.mean(axis=0), atol=1e-6)


def test_monomer_max_force_magnitudes():
    forces = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    offsets = np.array([0, 3, 6], dtype=int)
    m = monomer_max_force_magnitudes(forces, offsets)
    assert m.shape == (2,)
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(3.0)
