import numpy as np
import pytest

from mmml.md.interactions import InteractionPolicy, compile_interaction_policy
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
