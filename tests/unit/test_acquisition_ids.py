"""Stable structure IDs are geometry hashes, not source-path accidents."""

from __future__ import annotations

import numpy as np

from mmml.acquisition.ids import (
    composition_key,
    condition_key,
    geometry_fingerprint,
    group_key,
    stratum_key,
    structure_id,
)


def test_composition_hill_order():
    z = np.array([1, 1, 8])
    assert composition_key(z) == "H2O1"
    zc = np.array([6, 1, 1, 1, 1])
    assert composition_key(zc) == "C1H4"


def test_structure_id_is_stable_under_com_shift_and_noise_below_rounding():
    z = np.array([8, 1, 1])
    r = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    a = structure_id(r, z)
    b = structure_id(r + 1.7, z)
    c = structure_id(r + 4e-5, z)
    assert a == b == c


def test_structure_id_changes_when_geometry_changes():
    z = np.array([8, 1, 1])
    r = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    r2 = r.copy()
    r2[1, 0] += 0.05
    assert structure_id(r, z) != structure_id(r2, z)


def test_geometry_fingerprint_includes_cell():
    z = np.array([6])
    r = np.array([[0.0, 0.0, 0.0]])
    a = geometry_fingerprint(r, z, cell=np.diag([10.0, 10.0, 10.0]))
    b = geometry_fingerprint(r, z, cell=np.diag([12.0, 10.0, 10.0]))
    assert a != b


def test_stratum_joins_composition_and_condition():
    z = np.array([8, 1, 1])
    key = stratum_key(z, record={"temperature": 300.0, "phase": "liquid"})
    assert key.startswith("H2O1|")
    assert "T300" in key
    assert "phase-liquid" in key


def test_group_key_prefers_trajectory_identity():
    assert group_key({"group_seed": 7}, 99) == "group_seed:7"
    assert group_key({}, 3) == "ungrouped:3"


def test_condition_key_unspecified_when_empty():
    assert condition_key(None) == "unspecified"
    assert condition_key({}) == "unspecified"
