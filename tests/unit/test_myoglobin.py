"""Sperm-whale MbCO CRD, PHEM sites, and the His93 link atom. No CHARMM."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from argparse import Namespace

from mmml.interfaces.calculators.link_atoms import link_atom_position
from mmml.interfaces.pycharmmInterface.heme_electronic import (
    expand_counterions,
    resolve_metatomic_electronic_state,
)
from mmml.interfaces.pycharmmInterface.heme_library import (
    residues_from_cluster_args,
    topology_family,
)
from mmml.interfaces.pycharmmInterface.myoglobin import (
    HIS93_ML_CHARGE,
    HIS93_RESID,
    HIS93_SEGID,
    ION_SEGID,
    MBCO_SPIN_MULTIPLICITY,
    PHEM_SITES,
    crd_coordinate_table,
    his93_partition_columns,
    load_mbco,
    neutralize_mbco,
    sequence_read_command,
)


def _columns(structure):
    atoms = structure.atoms
    return (
        [atom.name for atom in atoms],
        [atom.resname for atom in atoms],
        [atom.resid for atom in atoms],
        [atom.segid for atom in atoms],
    )


def test_mbco_crd_keeps_the_crystal_protonation_and_drops_sulfate() -> None:
    structure = load_mbco()
    assert len(structure.atoms) == 3547
    kinds = {segment.segid: segment.kind for segment in structure.segments}
    assert kinds == {
        "MB": "protein",
        "HEM": "heme",
        "CO": "ligand",
        "WAT": "water",
        "XTLW": "water",
    }
    protein = next(segment for segment in structure.segments if segment.segid == "MB")
    assert protein.resnames[0] == "VAL"
    assert "HSD" in protein.resnames
    assert "HSE" in protein.resnames
    assert "HSP" in protein.resnames
    assert len(protein.resnames) == 153
    heme = next(segment for segment in structure.segments if segment.segid == "HEM")
    assert heme.resnames == ("HEME",)
    co = next(segment for segment in structure.segments if segment.segid == "CO")
    assert co.resnames == ("CO",)
    waters = [
        segment for segment in structure.segments if segment.kind == "water"
    ]
    assert sum(len(segment.resnames) for segment in waters) == 337
    assert all(atom.resname != "SO4" for atom in structure.atoms)
    assert PHEM_SITES == "MB 93 HEM 1"
    wat_resids = [atom.resid for atom in structure.atoms if atom.segid == "WAT"]
    assert 79 not in wat_resids
    assert max(wat_resids) == 213


def test_mbco_stays_one_monomer_and_ml_z_matches_the_selection() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        ml_z_aligned_to_selection,
        reconcile_n_monomers_with_psf,
    )

    z = np.arange(3547, dtype=int)
    args = Namespace(
        residue="MBCO",
        composition=None,
        n_molecules=1,
        _cluster_atoms_per_list=[3547],
        quiet=True,
    )
    n_mol, per = reconcile_n_monomers_with_psf(args, z, 1)
    assert n_mol == 1
    assert per == [3547]

    class _Sel:
        def get_atom_indexes(self):
            return [2, 0, 5]

    assert ml_z_aligned_to_selection([10, 11, 12, 13, 14, 15], _Sel()) == [12, 10, 15]
    assert ml_z_aligned_to_selection([12, 10, 15], _Sel()) == [12, 10, 15]


def test_water_coordinates_follow_segment_order_not_the_crystal_resid() -> None:
    structure = load_mbco()
    table = crd_coordinate_table(structure)
    assert len(table) == len(structure.atoms)
    for ordinal in range(1, 201):
        assert ("WAT", ordinal, "OH2") in table
        assert ("WAT", ordinal, "H1") in table
        assert ("WAT", ordinal, "H2") in table
    assert ("WAT", 213, "OH2") not in table
    assert ("MB", 93, "NE2") in table
    assert ("HEM", 1, "FE") in table


def test_proximal_histidine_is_his93_and_the_link_caps_the_imidazole() -> None:
    structure = load_mbco()
    atoms = structure.atoms
    his = [
        atom
        for atom in atoms
        if atom.segid == HIS93_SEGID and atom.resid == HIS93_RESID
    ]
    assert {atom.resname for atom in his} == {"HSD"}
    fe = next(atom for atom in atoms if atom.resname == "HEME" and atom.name == "FE")
    ne2 = next(atom for atom in his if atom.name == "NE2")
    distance = float(np.linalg.norm(np.subtract(fe.xyz, ne2.xyz)))
    assert distance == pytest.approx(2.19, abs=0.05)
    assert structure.formal_charge() == 2

    names, resnames, resids, segids = _columns(structure)
    ml, links = his93_partition_columns(names, resnames, resids, segids)
    ml_set = set(int(i) for i in ml)
    by_key = {
        (atom.segid, atom.resid, atom.name): i for i, atom in enumerate(atoms)
    }
    assert by_key[("HEM", 1, "FE")] in ml_set
    assert by_key[("CO", 1, "C")] in ml_set
    assert by_key[("CO", 1, "O")] in ml_set
    cg = by_key[("MB", 93, "CG")]
    cb = by_key[("MB", 93, "CB")]
    ca = by_key[("MB", 93, "CA")]
    assert cg in ml_set
    assert cb not in ml_set
    assert ca not in ml_set
    assert len(links) == 1
    assert links[0].qm_index == cg
    assert links[0].mm_index == cb
    pos = structure.positions()
    ghost = link_atom_position(pos[cg], pos[cb], links[0].bond_length_A)
    assert float(np.linalg.norm(ghost - pos[cg])) == pytest.approx(1.09)
    # The protein is one monomer wider than the 30 Å small-molecule cap.
    span = pos.max(axis=0) - pos.min(axis=0)
    assert float(np.linalg.norm(span)) > 30.0


def test_mbco_electronic_state_is_the_liganded_singlet() -> None:
    structure = load_mbco()
    charge = structure.formal_charge()
    bare = Namespace(residue="MBCO", n_molecules=1, composition=None, mm_region="none")
    state = resolve_metatomic_electronic_state(bare)
    assert state.charge == charge
    assert state.spin_multiplicity == MBCO_SPIN_MULTIPLICITY
    capped = Namespace(
        residue="MBCO",
        n_molecules=1,
        composition=None,
        mm_region="his93",
    )
    capped_state = resolve_metatomic_electronic_state(capped)
    assert capped_state.charge == HIS93_ML_CHARGE
    assert capped_state.spin_multiplicity == 1
    assert "singlet" in capped_state.reason
    with pytest.raises(ValueError, match="his93"):
        resolve_metatomic_electronic_state(
            Namespace(
                residue="MBCO",
                n_molecules=1,
                composition=None,
                mm_region="propionates",
            )
        )


def test_charmm_water_cube_is_4985_tip3_at_charge_plus_one() -> None:
    from mmml.interfaces.pycharmmInterface.charmm_paths import mmml_repo_root

    path = mmml_repo_root() / "setup/charmm/test/cbenchtest/mbco/mbco4985w.crd"
    structure = load_mbco(path)
    assert len(structure.atoms) == 17491
    kinds = {segment.segid: segment.kind for segment in structure.segments}
    assert kinds == {"MB": "protein", "HEM": "heme", "CO": "ligand", "WAT": "water"}
    waters = next(segment for segment in structure.segments if segment.segid == "WAT")
    assert len(waters.resnames) == 4985
    assert set(waters.resnames) == {"TIP3"}
    protein = next(segment for segment in structure.segments if segment.segid == "MB")
    assert protein.resnames[121] == "ASN"
    assert "HSE" not in protein.resnames
    assert "HSP" not in protein.resnames
    assert structure.formal_charge() == 1
    his = [
        atom
        for atom in structure.atoms
        if atom.segid == HIS93_SEGID and atom.resid == HIS93_RESID
    ]
    assert {atom.resname for atom in his} == {"HSD"}
    names, resnames, resids, segids = _columns(structure)
    ml, links = his93_partition_columns(names, resnames, resids, segids)
    assert len(ml) == 83
    assert len(links) == 1
    span = structure.positions().max(axis=0) - structure.positions().min(axis=0)
    assert float(np.linalg.norm(span)) == pytest.approx(98.41, abs=0.05)

    example = yaml.safe_load(
        (mmml_repo_root() / "examples/pet_omol_heme/yaml/mbco_nve.yaml").read_text()
    )
    assert example["mbco_crd"].endswith("mbco4985w.crd")
    assert float(example["box_size"]) == pytest.approx(55.49456)
    assert float(example["dynamics_max_monomer_extent"]) >= float(np.linalg.norm(span))


def test_periodic_water_cube_replaces_one_distant_tip3_with_chloride() -> None:
    assert sequence_read_command(("TIP3",) * 4985) == "read sequ TIP3 4985"
    assert sequence_read_command(("VAL", "LEU")) is None
    from mmml.interfaces.pycharmmInterface.charmm_paths import mmml_repo_root

    neutral = neutralize_mbco(
        load_mbco(mmml_repo_root() / "setup/charmm/test/cbenchtest/mbco/mbco4985w.crd")
    )
    assert neutral.formal_charge() == 0
    assert len(neutral.atoms) == 17489
    ions = [atom for atom in neutral.atoms if atom.resname == "CLA"]
    assert len(ions) == 1
    assert ions[0].segid == ION_SEGID
    assert ions[0].name == "CLA"
    waters = [segment for segment in neutral.segments if segment.kind == "water"]
    assert sum(len(segment.resnames) for segment in waters) == 4984
    assert sequence_read_command(waters[0].resnames) == "read sequ TIP3 4984"
    assert {segment.kind for segment in neutral.segments if segment.segid == ION_SEGID} == {
        "ion"
    }
    names, resnames, resids, segids = _columns(neutral)
    ml, links = his93_partition_columns(names, resnames, resids, segids)
    assert len(ml) == 83
    assert len(links) == 1
    solute = np.asarray(
        [atom.xyz for atom in neutral.atoms if atom.resname not in {"TIP3", "CLA"}],
        dtype=float,
    )
    distance = float(np.linalg.norm(solute - np.asarray(ions[0].xyz), axis=1).min())
    assert distance > 3.0
    periodic = Namespace(
        residue="MBCO",
        n_molecules=1,
        composition=None,
        mm_region="none",
        box_size=55.49456,
    )
    state = resolve_metatomic_electronic_state(periodic)
    assert state.charge == 0
    assert state.spin_multiplicity == 1


def test_mbco_topology_includes_water_and_rejects_counterions() -> None:
    args = Namespace(residue="MBCO", n_molecules=1, composition=None, mbco_crd=None)
    names = residues_from_cluster_args(args)
    assert "HEME" in names
    assert "CO" in names
    assert "TIP3" in names
    assert "HSE" in names
    assert "HSP" in names
    assert "SO4" not in names
    assert topology_family(names) == "heme"
    with pytest.raises(ValueError, match="counterions"):
        expand_counterions(
            Namespace(
                residue="MBCO",
                n_molecules=1,
                composition=None,
                counterions="SOD",
            )
        )
