"""Tests for lowering the two front-end configs into one RunConfig."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mmml.md.lowering import (
    ensemble_from_setup,
    runconfig_from_cg_config,
    runconfig_from_md_system_args,
    terms_from_cg_config,
)


# --- cg toggle -> term selection (doc §8) -----------------------------------


def test_terms_mm_only_intramolecular_ml():
    # default: ML intramolecular on, peptide-water handled by MM
    terms = terms_from_cg_config({})
    assert terms == ("ml_intra", "mm_nonbonded")


def test_terms_peptide_water_ml_adds_dimer_and_core():
    terms = terms_from_cg_config(
        {"peptide_water_ml": True, "peptide_water_ml_core_vdw": True}
    )
    assert terms == ("ml_intra", "mm_nonbonded", "ml_pep_water", "vdw_core")


def test_terms_no_ml_intra_and_biases():
    terms = terms_from_cg_config(
        {"use_ml_intramolecular": False, "constrain_phi_psi": True, "smd_enable": True}
    )
    assert terms == ("mm_nonbonded", "dihedral", "smd")


# --- cg config -> RunConfig --------------------------------------------------


def test_runconfig_from_cg_config_phases():
    cfg = {
        "checkpoint": "examples/ckpt.json",
        "n_waters": 100,
        "box_size": 30.0,
        "seed": 42,
        "temperature": 248.0,
        "dt_fs": 0.5,
        "nve_total_steps": 2000,
        "nvt_total_steps": 1000,
        "fire_steps": 500,
        "output_dir": "artifacts/cg",
    }
    nve = runconfig_from_cg_config(cfg, phase="nve")
    assert nve.ensemble.ensemble == "nve"
    assert nve.ensemble.n_steps == 2000
    assert nve.ensemble.temperature_K == 248.0
    assert nve.system.builder == "peptide_water"
    assert nve.system.n_molecules == 100
    assert nve.system.box_size == 30.0
    assert nve.backend == "jaxmd"
    assert nve.checkpoint == Path("examples/ckpt.json")
    assert nve.terms == ("ml_intra", "mm_nonbonded")

    fire = runconfig_from_cg_config(cfg, phase="fire")
    assert fire.ensemble.ensemble == "min"
    assert fire.ensemble.n_steps == 500

    nvt = runconfig_from_cg_config(cfg, phase="nvt")
    assert nvt.ensemble.ensemble == "nvt"
    assert nvt.ensemble.n_steps == 1000


def test_runconfig_from_cg_config_rejects_bad_phase():
    with pytest.raises(ValueError, match="unknown cg phase"):
        runconfig_from_cg_config({}, phase="npt")


# --- md-system setup parsing -------------------------------------------------


@pytest.mark.parametrize(
    "setup,expected",
    [
        ("pbc_nve", ("pbc", "nve")),
        ("free_nvt", ("free", "nvt")),
        ("pbc_npt", ("pbc", "npt")),
        ("pbc_thermalize", ("pbc", "nvt")),  # thermalize aliases to nvt
        ("free_thermalize", ("free", "nvt")),
    ],
)
def test_ensemble_from_setup(setup, expected):
    assert ensemble_from_setup(setup) == expected


def test_ensemble_from_setup_rejects_unknown():
    with pytest.raises(ValueError):
        ensemble_from_setup("pbc_minimize")
    with pytest.raises(ValueError):
        ensemble_from_setup("weird")


# --- md-system args -> RunConfig ---------------------------------------------


def test_runconfig_from_md_system_args():
    args = argparse.Namespace(
        setup="pbc_npt",
        dt_fs=1.0,
        ps=2.0,  # 2 ps at 1 fs = 2000 steps
        temperature=300.0,
        pressure=1.0,
        composition="DCM:10,TIP3:20",
        n_molecules=None,
        box_size=25.0,
        builder="packmol",
        seed=7,
        checkpoint="examples/ckpt.json",
        output_dir="artifacts/liquid",
    )
    cfg = runconfig_from_md_system_args(args)
    assert cfg.ensemble.ensemble == "npt"
    assert cfg.ensemble.space == "pbc"
    assert cfg.ensemble.n_steps == 2000
    assert cfg.ensemble.pressure_bar == 1.0
    assert cfg.system.builder == "packmol"
    assert cfg.system.composition == "DCM:10,TIP3:20"
    assert cfg.system.box_size == 25.0
    assert cfg.terms == ("ml_intra", "mm_nonbonded")
    assert cfg.checkpoint == Path("examples/ckpt.json")
    assert cfg.seed == 7


def test_runconfig_from_md_system_args_default_builder_and_terms():
    args = argparse.Namespace(setup="free_nve", dt_fs=0.5, ps=1.0, seed=0)
    cfg = runconfig_from_md_system_args(args)
    assert cfg.system.builder == "packmol"  # default when unset
    assert cfg.ensemble.space == "free"
    assert cfg.ensemble.n_steps == 2000  # 1 ps / 0.5 fs
    assert cfg.terms == ("ml_intra", "mm_nonbonded")
