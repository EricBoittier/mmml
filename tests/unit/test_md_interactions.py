import json
from pathlib import Path

import numpy as np
import pytest

from mmml.md.interactions import (
    InteractionPolicy,
    assert_interaction_plan_lowerable,
    compile_interaction_policy,
    interaction_plan_is_lowerable,
    interaction_policy_content_hash,
    load_interaction_policy,
    policy_is_lowerable,
)
from mmml.md.system import MolecularSystem


def _system(names=("PEP", "TIP3", "SOD")):
    groups = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5])]
    return MolecularSystem(
        R=np.zeros((6, 3)),
        Z=np.array([6, 1, 8, 1, 1, 11]),
        box=np.eye(3) * 20,
        mol_id=np.array([0, 0, 1, 1, 1, 2]),
        monomer_indices=groups,
        metadata={"residue_names": names},
    )


def _policy(**changes):
    data = {
        "schema_version": 1,
        "providers": {
            "pep_ml": {"kind": "ml", "checkpoint": "pep.pkl"},
            "pair_ml": {"kind": "ml", "checkpoint": "dimer.pkl"},
            "charmm": {"kind": "mm", "calculator": "cgenff"},
        },
        "monomers": {"PEP": "pep_ml", "TIP3": "charmm", "SOD": "charmm"},
        "pairs": [
            {
                "species": "PEP+TIP3",
                "near_provider": "pair_ml",
                "far_provider": "charmm",
                "switch": {"start_A": 5.0, "stop_A": 7.0},
            },
            {"species": "SOD+*", "provider": "charmm"},
            {"species": "*+*", "provider": "charmm"},
        ],
    }
    data.update(changes)
    return InteractionPolicy.from_mapping(data)


def _single_provider_policy():
    return InteractionPolicy.from_mapping(
        {
            "schema_version": 1,
            "providers": {"charmm": {"kind": "mm", "calculator": "cgenff"}},
            "monomers": {"PEP": "charmm", "TIP3": "charmm", "SOD": "charmm"},
            "pairs": [{"species": "*+*", "provider": "charmm"}],
        }
    )


def test_compiler_assigns_every_monomer_and_pair_once():
    plan = compile_interaction_policy(_system(), _policy())
    assert len(plan.monomers) == 3
    assert len(plan.pairs) == 3
    pep_water = next(p for p in plan.pairs if set(p.species) == {"PEP", "TIP3"})
    assert pep_water.near_provider == "pair_ml"
    assert pep_water.far_provider == "charmm"
    assert pep_water.switch.stop_A == 7.0
    assert next(p for p in plan.pairs if "SOD" in p.species).provider == "charmm"


def test_compiler_rejects_uncovered_species():
    policy = _policy(monomers={"PEP": "pep_ml", "TIP3": "charmm"})
    with pytest.raises(ValueError, match="no monomer provider.*SOD"):
        compile_interaction_policy(_system(), policy)


def test_compiler_rejects_uncovered_pair():
    policy = _policy(pairs=[{"species": "PEP+TIP3", "provider": "pair_ml"}])
    with pytest.raises(ValueError, match="no pair provider"):
        compile_interaction_policy(_system(), policy)


def test_compiler_rejects_equal_specificity_ambiguity():
    policy = _policy(pairs=[
        {"species": "PEP+*", "provider": "pair_ml"},
        {"species": "*+TIP3", "provider": "charmm"},
        {"species": "*+*", "provider": "charmm"},
    ])
    with pytest.raises(ValueError, match="ambiguous pair rules"):
        compile_interaction_policy(_system(), policy)


def test_policy_rejects_unknown_provider_and_schema():
    with pytest.raises(ValueError, match="undefined providers"):
        _policy(monomers={"PEP": "missing", "TIP3": "charmm", "SOD": "charmm"})
    with pytest.raises(ValueError, match="schema_version"):
        _policy(schema_version=99)


def test_topology_labels_are_mandatory_and_aligned():
    with pytest.raises(ValueError, match="aligned"):
        compile_interaction_policy(_system(names=("PEP",)), _policy())


def test_policy_json_round_trip(tmp_path):
    policy = _policy()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.to_mapping()), encoding="utf-8")
    assert load_interaction_policy(path).to_mapping() == policy.to_mapping()


def test_single_provider_is_lowerable_multi_is_not():
    assert policy_is_lowerable(_single_provider_policy())
    assert not policy_is_lowerable(_policy())
    plan_ok = compile_interaction_policy(_system(), _single_provider_policy())
    assert interaction_plan_is_lowerable(plan_ok)
    assert_interaction_plan_lowerable(plan_ok)
    plan_bad = compile_interaction_policy(_system(), _policy())
    with pytest.raises(NotImplementedError, match="not yet lowerable"):
        assert_interaction_plan_lowerable(plan_bad)


def test_policy_content_hash_stable():
    a = interaction_policy_content_hash(_single_provider_policy())
    b = interaction_policy_content_hash(_single_provider_policy())
    assert a == b
    assert len(a) == 64


def test_config_relative_interaction_policy_resolution(tmp_path):
    from mmml.cli.run.md_config import resolve_config_relative_path

    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    policy = cfg_dir / "policy.yaml"
    policy.write_text(
        "schema_version: 1\n"
        "providers: {cgenff: {kind: mm, calculator: cgenff}}\n"
        "monomers: {DCM: cgenff}\n"
        "pairs: [{species: ['*', '*'], provider: cgenff}]\n",
        encoding="utf-8",
    )
    cfg = cfg_dir / "md.yaml"
    cfg.write_text("interaction_policy: policy.yaml\n", encoding="utf-8")
    resolved = resolve_config_relative_path(cfg, "policy.yaml")
    assert resolved == policy.resolve()
    loaded = load_interaction_policy(resolved)
    assert policy_is_lowerable(loaded)


def test_md_system_parse_resolves_interaction_policy(tmp_path, monkeypatch):
    from mmml.cli.run import md_system as md_system_mod

    cfg_dir = tmp_path / "run"
    cfg_dir.mkdir()
    policy = cfg_dir / "interaction_policy.yaml"
    policy.write_text(
        "schema_version: 1\n"
        "providers: {cgenff: {kind: mm, calculator: cgenff}}\n"
        "monomers: {DCM: cgenff}\n"
        "pairs: [{species: ['*', '*'], provider: cgenff}]\n",
        encoding="utf-8",
    )
    cfg = cfg_dir / "md.yaml"
    cfg.write_text(
        "composition: 'DCM:2'\n"
        "interaction_policy: interaction_policy.yaml\n"
        "checkpoint: /tmp/dummy.json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.sync_spatial_mpi_env_from_args",
        lambda args: None,
    )
    args = md_system_mod.parse_md_system_args(["--config", str(cfg)])
    assert Path(args.interaction_policy).resolve() == policy.resolve()


def test_validate_and_record_fail_closed_multi_provider(tmp_path):
    import argparse

    from mmml.cli.run.md_system import _validate_and_record_interaction_policy

    policy = tmp_path / "multi.yaml"
    policy.write_text(
        "schema_version: 1\n"
        "providers:\n"
        "  pep_ml: {kind: ml, checkpoint: pep.pkl}\n"
        "  charmm: {kind: mm, calculator: cgenff}\n"
        "monomers: {PEP: pep_ml, TIP3: charmm}\n"
        "pairs:\n"
        "  - {species: ['*', '*'], provider: charmm}\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(interaction_policy=policy, quiet=True)
    with pytest.raises(NotImplementedError, match="not yet lowerable"):
        _validate_and_record_interaction_policy(args)
