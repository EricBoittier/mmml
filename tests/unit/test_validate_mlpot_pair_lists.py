"""Unit tests for validate_mlpot_pair_lists (no CRD/DCD files)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.validate_mlpot_pair_lists import analyze_pair_lists  # noqa: E402


def test_free_space_all_dimer_slots_no_sparse_truncation() -> None:
    n_mol = 8
    per = 5
    pos = np.zeros((n_mol * per, 3), dtype=float)
    for mi in range(n_mol):
        pos[mi * per : (mi + 1) * per, 0] = float(mi) * 8.0
    report = analyze_pair_lists(pos, n_mol, per, free_space=True)
    assert report["n_dimers_total"] == 28
    assert report["ml_max_active_dimers_cap"] == 28
    assert report["sparse_ml_active"] is False
    assert all(d["ml_evaluated"] for d in report["dimers"])


def test_handoff_zone_detected_for_mid_range_com() -> None:
    pos = np.zeros((10, 3), dtype=float)
    pos[0:5, :] = 0.0
    pos[5:10, 0] = 5.45
    report = analyze_pair_lists(pos, 2, 5, mm_switch_on=5.5, ml_switch_width=0.1)
    assert report["dimers"][0]["zone"] == "handoff"


def test_close_contact_flagged() -> None:
    pos = np.zeros((10, 3), dtype=float)
    pos[5, 0] = 0.5
    report = analyze_pair_lists(pos, 2, 5)
    assert report["close_contacts_under_1A"]
    assert report["close_contacts_under_1A"][0]["min_inter_atom_A"] < 1.0
