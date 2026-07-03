"""Unit tests for trialanine NB parity helpers (no PyCHARMM)."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.trialanine_nb_parity import (
    CategoryNonbondedTotals,
    TopPairRecord,
    _aggregate_by_category,
    _top_pairs,
    classify_pair_categories,
    classify_pair_category,
)


def test_classify_pair_category() -> None:
    n_pep = 42
    assert classify_pair_category(0, 1, n_pep) == "pep_pep"
    assert classify_pair_category(0, 50, n_pep) == "pep_water"
    assert classify_pair_category(50, 60, n_pep) == "water_water"


def test_classify_pair_categories_vectorized() -> None:
    n_pep = 5
    pi = np.array([0, 0, 4, 5, 6], dtype=np.int32)
    pj = np.array([1, 5, 5, 6, 7], dtype=np.int32)
    cats = classify_pair_categories(pi, pj, n_pep)
    assert list(cats) == ["pep_pep", "pep_water", "pep_water", "water_water", "water_water"]


class _FakeDecomp:
    def __init__(self) -> None:
        self.pair_i = np.array([0, 0, 5], dtype=np.int32)
        self.pair_j = np.array([1, 5, 6], dtype=np.int32)
        self.r_A = np.array([3.0, 4.0, 5.0])
        self.vdw_kcal = np.array([10.0, -2.0, 0.5])
        self.elec_kcal = np.array([-1.0, -3.0, -0.1])


def test_aggregate_by_category() -> None:
    decomp = _FakeDecomp()
    cats = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_peptide_atoms=5)
    rows = _aggregate_by_category(decomp, cats)
    assert len(rows) == 3
    pp = next(r for r in rows if r.category == "pep_pep")
    assert pp.n_pairs == 1
    assert pp.vdw_kcal == 10.0
    assert pp.elec_kcal == -1.0
    pw = next(r for r in rows if r.category == "pep_water")
    assert pw.n_pairs == 1
    assert pw.vdw_kcal == -2.0


def test_top_pairs_ranks_by_abs_vdw() -> None:
    decomp = _FakeDecomp()
    cats = classify_pair_categories(decomp.pair_i, decomp.pair_j, n_peptide_atoms=5)
    top = _top_pairs(decomp, cats, term="vdw", n=2, category_filter="pep_pep")
    assert len(top) == 1
    assert isinstance(top[0], TopPairRecord)
    assert top[0].atom_i == 1
    assert top[0].vdw_kcal == 10.0


def test_category_nonbonded_totals_total() -> None:
    row = CategoryNonbondedTotals("pep_pep", 3, 1.5, -2.0, 4.0)
    assert row.total_kcal == -0.5
