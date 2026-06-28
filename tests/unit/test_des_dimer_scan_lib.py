"""Unit tests for des_dimer_pair_scans pair matrix."""

from __future__ import annotations

import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows" / "des_dimer_pair_scans"
sys.path.insert(0, str(WORKFLOW / "scripts"))

import scan_lib as sl  # noqa: E402


def test_species_count():
    cfg = sl.load_config(WORKFLOW / "config.yaml")
    species = sl.iter_species(cfg)
    assert len(species) == 12


def test_pair_count():
    cfg = sl.load_config(WORKFLOW / "config.yaml")
    pairs = list(sl.iter_pairs(cfg))
    assert len(pairs) == 78  # C(12,2)+12 homo


def test_homo_composition():
    cfg = sl.load_config(WORKFLOW / "config.yaml")
    aco = next(s for s in sl.iter_species(cfg) if s.tag == "aco")
    assert sl.composition_for_pair(aco, aco) == "ACO:2"


def test_hetero_composition():
    cfg = sl.load_config(WORKFLOW / "config.yaml")
    aco = next(s for s in sl.iter_species(cfg) if s.tag == "aco")
    meoh = next(s for s in sl.iter_species(cfg) if s.tag == "meoh")
    assert sl.composition_for_pair(aco, meoh) == "ACO:1,MEOH:1"
    assert sl.pair_tag(aco, meoh) == "aco__meoh"


def test_cgenff_residues_present():
    cfg = sl.load_config(WORKFLOW / "config.yaml")
    missing = sl.validate_species_residues(cfg)
    assert missing == []
