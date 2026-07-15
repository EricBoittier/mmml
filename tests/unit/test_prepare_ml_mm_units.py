"""Regression tests for the CGenFF MM baseline in prepare_ml_mm_dataset.

The inter-monomer MM energy is stored in eV. A previously inverted unit constant
(``1/kcal * mol`` = 23.06, the eV -> kcal/mol factor) inflated ``E_cgenff_mm`` and
``F_cgenff_mm`` by 531.8x, so a well-separated dimer came out at -276 kcal/mol
instead of -0.5 kcal/mol.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_ml_mm_dataset.py"


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prepare_ml_mm_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kcal_to_ev_constant(prep):
    assert prep.KCAL_TO_EV == pytest.approx(0.0433641153, rel=1e-6)


def test_inter_monomer_energy_matches_hand_computed_lj_plus_coulomb(prep):
    """Two neutral LJ sites 3.5 A apart: energy must equal the textbook expression, in eV."""
    sigmas, epsilons = prep._CGENFF_SIGMAS, prep._CGENFF_EPSILONS
    t_a = np.array([prep._NB_MAP["OT"]])
    t_b = np.array([prep._NB_MAP["OT"]])
    q_a = np.array([0.4])
    q_b = np.array([-0.4])
    r = 3.5
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])

    e_ev, forces = prep.compute_inter_monomer_cgenff_mm_fast(
        pos, [0], t_a, q_a, [1], t_b, q_b
    )

    sig = 0.5 * (sigmas[t_a[0]] + sigmas[t_b[0]])
    eps = np.sqrt(epsilons[t_a[0]] * epsilons[t_b[0]])
    sr6 = (sig / r) ** 6
    e_kcal = 4.0 * eps * (sr6**2 - sr6) + prep.K_COULOMB_KCAL_ANG * q_a[0] * q_b[0] / r

    assert e_ev == pytest.approx(e_kcal * prep.KCAL_TO_EV, rel=1e-9)
    assert forces.shape == pos.shape
    # Newton's third law: inter-monomer forces must sum to zero.
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-12)


def test_separated_dimer_energy_is_physical(prep):
    """A well-separated neutral dimer must be O(1) kcal/mol, not O(100)."""
    t = np.array([prep._NB_MAP["OT"]])
    pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    e_ev, _ = prep.compute_inter_monomer_cgenff_mm_fast(
        pos, [0], t, np.array([0.0]), [1], t, np.array([0.0])
    )
    assert abs(e_ev / 0.0433641153) < 5.0


def test_only_inter_monomer_pairs_contribute(prep):
    """Intramolecular geometry must not change the inter-monomer MM energy."""
    t2 = np.array([prep._NB_MAP["OT"], prep._NB_MAP["HT"]])
    q2 = np.array([-0.4, 0.4])
    t1 = np.array([prep._NB_MAP["OT"]])
    q1 = np.array([0.0])

    # Monomer A is two atoms; move its internal bond length but keep both atoms'
    # positions relative to B fixed by placing B far along +z.
    pos_close = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 30.0]])
    pos_far = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 30.0]])

    e_close, _ = prep.compute_inter_monomer_cgenff_mm_fast(
        pos_close, [0, 1], t2, q2, [2], t1, q1
    )
    e_far, _ = prep.compute_inter_monomer_cgenff_mm_fast(
        pos_far, [0, 1], t2, q2, [2], t1, q1
    )
    # B is neutral and 30 A away: both must be ~0, and no intramolecular A-A term
    # (which would be enormous at 0.9 A) may leak in.
    assert abs(e_close / 0.0433641153) < 0.1
    assert abs(e_far / 0.0433641153) < 0.1
