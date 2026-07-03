"""Unit tests for neighbor list snapshot utilities (no PyCHARMM)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.utils.neighbor_list_snapshot import (
    capture_mmml_inter_monomer_pairs,
    compare_snapshots,
    cubic_cell_matrix,
    save_neighbor_list_artifacts,
    uniform_monomer_offsets,
)


def _two_dimer_cluster(box_side: float = 40.0, separation: float = 8.0):
    n_monomers = 2
    apm = 5
    offsets = uniform_monomer_offsets(n_monomers, apm)
    n_atoms = n_monomers * apm
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    for mi in range(n_monomers):
        start = int(offsets[mi])
        com = np.array([separation * mi, 0.0, 0.0], dtype=np.float64)
        for k in range(apm):
            positions[start + k] = com + np.array([0.3 * k, 0.1 * (k % 2), 0.0])
    cell = cubic_cell_matrix(box_side)
    return positions, cell, offsets


def test_capture_mmml_inter_monomer_pairs_cell_list() -> None:
    positions, cell, offsets = _two_dimer_cluster()
    snap = capture_mmml_inter_monomer_pairs(
        positions=positions,
        cell=cell,
        cutoff_A=13.0,
        monomer_offsets=offsets,
        backend="cell_list",
    )
    assert snap.pairs
    assert snap.pairs[0].monomer_i != snap.pairs[0].monomer_j
    assert snap.pairs[0].distance_A > 0.0


def test_compare_snapshots_reports_diff() -> None:
    positions, cell, offsets = _two_dimer_cluster()
    left = capture_mmml_inter_monomer_pairs(
        positions=positions,
        cell=cell,
        cutoff_A=13.0,
        monomer_offsets=offsets,
        backend="cell_list",
    )
    right = capture_mmml_inter_monomer_pairs(
        positions=positions,
        cell=cell,
        cutoff_A=6.0,
        monomer_offsets=offsets,
        backend="cell_list",
    )
    cmp = compare_snapshots(left, right)
    assert cmp["n_left"] >= cmp["n_right"]
    assert cmp["n_only_left"] >= 0


def test_save_neighbor_list_artifacts_writes_json_and_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    positions, cell, offsets = _two_dimer_cluster()
    mmml = capture_mmml_inter_monomer_pairs(
        positions=positions,
        cell=cell,
        cutoff_A=13.0,
        monomer_offsets=offsets,
        backend="cell_list",
    )
    paths = save_neighbor_list_artifacts(
        tmp_path,
        positions=positions,
        cell=cell,
        monomer_offsets=offsets,
        mmml=mmml,
        extra_meta={"tag": "unit"},
    )
    assert paths["json"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["meta"]["tag"] == "unit"
    assert payload["mmml"]["pairs"]
    assert paths["plot"].is_file()
    assert paths["mmml_csv"].is_file()
