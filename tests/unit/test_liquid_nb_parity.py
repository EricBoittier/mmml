"""Unit tests for liquid NB parity helpers (no PyCHARMM)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.liquid_nb_parity import (
    CategoryNonbondedTotals,
    _aggregate_liquid_by_category,
    classify_liquid_pair_categories,
    classify_liquid_pair_category,
    diagnose_inter_monomer_vdw,
    monomer_id_from_offsets,
)


def test_monomer_id_from_offsets() -> None:
    offsets = np.array([0, 5, 10, 15], dtype=np.int32)
    mid = monomer_id_from_offsets(offsets, 15)
    assert list(mid) == [0] * 5 + [1] * 5 + [2] * 5


def test_classify_liquid_pair_category() -> None:
    mid = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    assert classify_liquid_pair_category(0, 1, mid) == "intra_monomer"
    assert classify_liquid_pair_category(0, 3, mid) == "inter_monomer"


def test_classify_liquid_pair_categories_vectorized() -> None:
    mid = np.array([0, 0, 1, 1], dtype=np.int32)
    pi = np.array([0, 0, 2], dtype=np.int32)
    pj = np.array([1, 2, 3], dtype=np.int32)
    cats = classify_liquid_pair_categories(pi, pj, mid)
    assert list(cats) == ["intra_monomer", "inter_monomer", "inter_monomer"]


class _FakeDecomp:
    def __init__(self) -> None:
        self.pair_i = np.array([0, 0, 2], dtype=np.int32)
        self.pair_j = np.array([1, 2, 3], dtype=np.int32)
        self.r_A = np.array([2.5, 4.0, 6.0])
        self.vdw_kcal = np.array([0.5, 3.0, -1.0])
        self.elec_kcal = np.array([-0.2, -1.0, 0.3])


def test_aggregate_liquid_by_category() -> None:
    decomp = _FakeDecomp()
    mid = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    cats = classify_liquid_pair_categories(decomp.pair_i, decomp.pair_j, mid)
    rows = _aggregate_liquid_by_category(decomp, cats)
    intra = next(r for r in rows if r.category == "intra_monomer")
    inter = next(r for r in rows if r.category == "inter_monomer")
    assert intra.n_pairs == 1
    assert intra.vdw_kcal == 0.5
    assert inter.n_pairs == 2
    assert inter.vdw_kcal == 2.0


def test_diagnose_inter_monomer_vdw() -> None:
    by_cat = (
        CategoryNonbondedTotals("intra_monomer", 10, 0.2, 0.0, 3.0),
        CategoryNonbondedTotals("inter_monomer", 100, 2.4, -1.0, 7.0),
    )
    diag = diagnose_inter_monomer_vdw(-1.0, by_cat)
    assert diag.charmm_implied_inter_vdw_kcal == pytest.approx(-1.2)
    assert diag.inter_vdw_delta_kcal == pytest.approx(3.6)
