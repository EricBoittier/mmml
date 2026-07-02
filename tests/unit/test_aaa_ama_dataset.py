"""Unit tests for ``aaa.ama`` dataset inspection (no download required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmml.data.external.aaa_ama import (
    AAA_DATASET_URL,
    inspect_dataset_aaa,
    load_dataset_aaa,
    per_element_force_magnitudes,
)


@pytest.fixture(scope="module")
def summary_path() -> Path:
    return Path(__file__).resolve().parents[2] / "mmml" / "data" / "external" / "aaa_ama_dataset_summary.json"


def test_bundled_summary_matches_expected_topology(summary_path: Path) -> None:
    if not summary_path.is_file():
        pytest.skip("run scripts/analyze_aaa_ama_dataset.py to generate summary")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["n_frames"] == 12500
    assert report["n_atoms"] == 34
    assert report["formula"] == "C9H18N3O4"
    assert report["net_charge"] == pytest.approx(1.0)
    assert "ACE" in report["molecule_label"]


def test_inspect_synthetic_npz_matches_aaa_layout() -> None:
    n_frames, n_atoms = 4, 34
    z0 = np.array(
        [7, 1, 1, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 8, 1],
        dtype=int,
    )
    data = {
        "N": np.full(n_frames, n_atoms, dtype=int),
        "Z": np.tile(z0, (n_frames, 1)),
        "R": np.random.default_rng(0).normal(size=(n_frames, n_atoms, 3)),
        "E": np.linspace(-132.0, -131.0, n_frames),
        "F": np.random.default_rng(1).normal(scale=0.5, size=(n_frames, n_atoms, 3)),
        "Q": np.ones(n_frames),
        "D": np.zeros((n_frames, 3)),
    }
    report = inspect_dataset_aaa(data)
    assert report.n_frames == n_frames
    assert report.n_atoms == n_atoms
    assert report.formula == "C9H18N3O4"
    assert len(report.element_species) == 4
    by_elem = per_element_force_magnitudes(data)
    assert set(by_elem) == {"H", "C", "N", "O"}
    assert by_elem["H"].size == n_frames * 18


@pytest.mark.skipif(
    not Path("/tmp/dataset_aaa.npz").is_file(),
    reason="optional: download dataset_aaa.npz to /tmp for live NPZ test",
)
def test_load_live_npz_if_present() -> None:
    data = load_dataset_aaa("/tmp/dataset_aaa.npz")
    report = inspect_dataset_aaa(data)
    assert report.source_url == AAA_DATASET_URL
    assert report.n_frames > 1000
