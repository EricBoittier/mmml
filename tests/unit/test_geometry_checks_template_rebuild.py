"""Peer-template rebuild for high-force monomers (jax-md FIRE worst case)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.utils.geometry_checks import (
    TEMPLATE_DONOR_IDEAL_TIP3,
    monomer_max_force_magnitudes,
    rebuild_high_force_monomers_from_peers,
    select_template_donor_monomer,
    tip3_hoh_angle_deg,
    tip3_peer_donor_acceptable,
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
    assert tip3_hoh_angle_deg(new_pos[3:6]) == pytest.approx(104.5, abs=1.0)
    # COM of rebuilt monomer preserved.
    assert np.allclose(new_pos[3:6].mean(axis=0), crushed.mean(axis=0), atol=1e-6)


def test_donor_prefers_healthy_geometry_over_softest_force():
    """Softest-by-force can still be crushed; do not copy it."""
    soft_crushed = _water([0.0, 0.0, 0.0], angle_deg=86.0)
    hard_healthy = _water([5.0, 0.0, 0.0], angle_deg=104.5)
    victim = _water([10.0, 0.0, 0.0], angle_deg=40.0)
    pos = np.vstack([soft_crushed, hard_healthy, victim])
    offsets = np.array([0, 3, 6, 9], dtype=int)
    forces = np.zeros_like(pos)
    forces[0:3] = 1.0  # softest but bad HOH
    forces[3:6] = 3.0  # harder but healthy geometry
    forces[6:9] = 10.0

    mol_f = monomer_max_force_magnitudes(forces, offsets)
    assert int(np.argmin(mol_f)) == 0
    donor = select_template_donor_monomer(pos, offsets, mol_f, size=3)
    assert donor == 1
    assert tip3_peer_donor_acceptable(pos[3:6])
    assert not tip3_peer_donor_acceptable(pos[0:3])

    new_pos, victims, donor_id = rebuild_high_force_monomers_from_peers(
        pos,
        forces,
        offsets,
        force_percentile=50.0,
        max_rebuild=2,
        min_force_eVA=1.0,
    )
    assert 2 in victims
    assert donor_id == 1
    assert tip3_hoh_angle_deg(new_pos[6:9]) == pytest.approx(104.5, abs=1.0)


def test_all_peers_crushed_falls_back_to_ideal_tip3():
    """When every water fails the HOH/OH gate, use bundled ideal TIP3."""
    a = _water([0.0, 0.0, 0.0], angle_deg=86.0)
    b = _water([5.0, 0.0, 0.0], angle_deg=84.0)
    c = _water([10.0, 0.0, 0.0], angle_deg=40.0)
    pos = np.vstack([a, b, c])
    offsets = np.array([0, 3, 6, 9], dtype=int)
    forces = np.ones_like(pos) * 7.0
    forces[6:9] = 10.0

    mol_f = monomer_max_force_magnitudes(forces, offsets)
    assert select_template_donor_monomer(pos, offsets, mol_f, size=3) == (
        TEMPLATE_DONOR_IDEAL_TIP3
    )

    new_pos, victims, donor = rebuild_high_force_monomers_from_peers(
        pos,
        forces,
        offsets,
        force_percentile=50.0,
        max_rebuild=3,
        min_force_eVA=1.0,
    )
    assert donor == TEMPLATE_DONOR_IDEAL_TIP3
    assert victims
    for mi in victims:
        s, e = int(offsets[mi]), int(offsets[mi + 1])
        assert tip3_peer_donor_acceptable(new_pos[s:e])


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
