"""YAML ML/MM cuts. No CHARMM."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from karml.cli.run.md_system import build_pycharmm_command, parse_md_system_args
from karml.interfaces.pycharmmInterface.charmm_paths import karml_repo_root
from karml.interfaces.pycharmmInterface.heme_electronic import (
    resolve_metatomic_electronic_state,
)
from karml.interfaces.pycharmmInterface.ml_cut import load_ml_cut, partition_ml_cut
from karml.interfaces.pycharmmInterface.mlpot.charmm_energy_policy import (
    resolve_charmm_energy_term_policies,
)
from karml.interfaces.pycharmmInterface.myoglobin import (
    his93_partition_columns,
    load_mbco,
)

HIS93_YAML = Path("examples/pet_omol_heme/yaml/his93.yaml")


def _columns(structure):
    atoms = structure.atoms
    return (
        [atom.name for atom in atoms],
        [atom.resname for atom in atoms],
        [atom.resid for atom in atoms],
        [atom.segid for atom in atoms],
    )


def test_his93_yaml_matches_the_preset_partition() -> None:
    structure = load_mbco()
    columns = _columns(structure)
    preset_ml, preset_links = his93_partition_columns(*columns)
    spec = load_ml_cut(karml_repo_root() / HIS93_YAML)
    ml, links = partition_ml_cut(spec, *columns)
    assert spec.charge == -2
    assert spec.spin_multiplicity == 1
    assert list(ml) == list(preset_ml)
    assert len(links) == 1
    assert links[0].qm_index == preset_links[0].qm_index
    assert links[0].mm_index == preset_links[0].mm_index


def test_mbco_yaml_points_at_the_cut_file_and_forwards_it() -> None:
    from karml.cli.run.md_pbc_suite import pycharmm_mlpot

    args = parse_md_system_args(
        ["--config", "examples/pet_omol_heme/yaml/mbco_nve.yaml"]
    )
    assert args.ml_cut == "examples/pet_omol_heme/yaml/his93.yaml"
    assert args.mm_region in (None, "none")
    cmd = build_pycharmm_command(args)
    assert cmd[cmd.index("--ml-cut") + 1] == args.ml_cut
    parsed = pycharmm_mlpot.parse_args(cmd)
    assert parsed.ml_cut == args.ml_cut


def test_cut_file_sets_charge_and_spin_and_keeps_protein_vdw() -> None:
    args = Namespace(
        residue="MBCO",
        composition=None,
        ml_cut=str(karml_repo_root() / HIS93_YAML),
        mm_region=None,
        charge=None,
        spin_multiplicity=None,
        box_size=55.5,
        mm_nonbond_mode="jax_mic",
        periodic_charmm_vdw=False,
        charmm_zero_energy_terms=None,
    )
    state = resolve_metatomic_electronic_state(args)
    assert state.charge == -2
    assert state.spin_multiplicity == 1
    assert "his93.yaml" in state.reason
    policies = resolve_charmm_energy_term_policies(args)
    assert [policy.name for policy in policies] == []


def test_ml_cut_rejects_mm_region(tmp_path: Path) -> None:
    from karml.interfaces.pycharmmInterface.ml_cut import ml_cut_from_args

    cut = tmp_path / "cut.yaml"
    cut.write_text(
        "charge: 0\nspin_multiplicity: 1\nml_atoms:\n  - resname: HEME\n",
        encoding="utf-8",
    )
    args = Namespace(
        ml_cut=str(cut),
        mm_region="his93",
        _cluster_atom_names=["FE"],
        _cluster_atom_resnames=["HEME"],
        _cluster_atom_resids=[1],
        _cluster_atom_segids=["HEM"],
    )
    with pytest.raises(ValueError, match="replaces --mm-region"):
        ml_cut_from_args(args)
