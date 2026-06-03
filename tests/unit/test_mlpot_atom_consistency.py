"""Unit tests for CHARMM ↔ MLpot atom identity helpers (no CHARMM runtime)."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.setup import _masses_consistent_with_z


def test_masses_consistent_with_z_accepts_matching_masses():
    import ase.data

    z = np.array([6, 1, 17], dtype=int)
    masses = ase.data.atomic_masses_common[z]
    assert _masses_consistent_with_z(masses, z) == []


def test_masses_consistent_with_z_accepts_cgenff_chlorine_mass():
    """CGenFF Cl mass (35.453) differs from ASE (34.969) but still maps to Z=17."""
    z = np.array([6, 1, 17, 17], dtype=int)
    masses = np.array([12.011, 1.008, 35.453, 35.453], dtype=float)
    assert _masses_consistent_with_z(masses, z) == []


def test_masses_consistent_with_z_flags_wrong_z_assignment():
    z = np.array([6, 1], dtype=int)
    masses = np.array([12.0, 5.0], dtype=float)
    issues = _masses_consistent_with_z(masses, z)
    assert len(issues) == 1
    assert "atom 1" in issues[0]
    assert "best matches" in issues[0]
