"""HEME charge/spin, carboxylate counterions, and propionate link atoms."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from karml.data.units import EV_TO_KCAL_MOL
from karml.interfaces.calculators.ase_fragment_hybrid import evaluate_whole_system
from karml.interfaces.calculators.link_atoms import (
    LinkAtom,
    link_atom_position,
    project_link_force,
)
from karml.interfaces.pycharmmInterface.cgenff_residues import require_cgenff_residue_name
from karml.interfaces.pycharmmInterface.heme_electronic import (
    HEME_FORMAL_CHARGE,
    HEME_SPIN_MULTIPLICITY,
    PROPIONATE_MM_NAMES,
    expand_counterions,
    propionate_partition,
    resolve_metatomic_electronic_state,
    seat_heme_counterions,
)
from karml.interfaces.pycharmmInterface.heme_library import (
    heme_reference_coordinate_table,
    heme_reference_positions,
    topology_family,
)
from karml.interfaces.pycharmmInterface.mlpot.metatomic_mlpot import (
    MetatomicMlpotCalculator,
    build_metatomic_mlpot_model,
)


def _heme_names() -> list[str]:
    return list(heme_reference_coordinate_table())


def test_bare_heme_is_charge_minus_two_triplet() -> None:
    state = resolve_metatomic_electronic_state(
        Namespace(residue="HEME", n_molecules=1, composition=None)
    )
    assert state.charge == HEME_FORMAL_CHARGE == -2
    assert state.spin_multiplicity == HEME_SPIN_MULTIPLICITY == 3


def test_counterions_neutralize_heme_and_keep_the_triplet() -> None:
    args = Namespace(residue="HEME", n_molecules=1, composition=None, counterions="SOD")
    expand_counterions(args)
    assert args.composition == "HEME:1,SOD:2"
    expand_counterions(args)
    assert args.composition == "HEME:1,SOD:2"
    state = resolve_metatomic_electronic_state(args)
    assert state.charge == 0
    assert state.spin_multiplicity == 3
    assert state.monomer_charges == (-2, 1, 1)
    assert state.monomer_spins == (3, 1, 1)
    assert require_cgenff_residue_name("SOD") == "SOD"
    assert topology_family(("HEME", "SOD")) == "heme"
    assert topology_family(("HEME", "TIP3")) == "heme"
    with pytest.raises(ValueError, match="toppar_all36_prot_heme"):
        topology_family(("HEME", "MEOH"))


def test_two_hemes_need_an_explicit_spin() -> None:
    with pytest.raises(ValueError, match="spin-multiplicity"):
        resolve_metatomic_electronic_state(
            Namespace(residue="HEME", n_molecules=2, composition=None)
        )


def test_sodiums_sit_on_the_two_carboxylates() -> None:
    names = _heme_names()
    heme = heme_reference_positions(names)
    assert heme is not None
    positions = np.vstack([heme, np.zeros((2, 3))])
    seated = seat_heme_counterions(
        positions,
        [*names, "SOD", "SOD"],
        ["HEME", "SOD", "SOD"],
        [len(names), 1, 1],
    )
    oxygens = {
        "A": np.stack([heme[names.index("O1A")], heme[names.index("O2A")]]),
        "D": np.stack([heme[names.index("O1D")], heme[names.index("O2D")]]),
    }
    for ion, group in zip(seated[-2:], ("A", "D")):
        dists = np.linalg.norm(oxygens[group] - ion, axis=1)
        assert dists == pytest.approx(2.35, abs=0.05)
        other = "D" if group == "A" else "A"
        assert np.linalg.norm(oxygens[other] - ion, axis=1).min() > 5.0
    assert np.linalg.norm(seated[-1] - seated[-2]) > 5.0


def test_propionate_cut_adds_two_ghost_hydrogens() -> None:
    names = _heme_names()
    ml, links = propionate_partition(names)
    assert len(links) == 2
    assert len(ml) == len(names) - len(PROPIONATE_MM_NAMES)
    assert names[links[0].qm_index] == "CAA"
    assert names[links[0].mm_index] == "CBA"
    assert names[links[1].qm_index] == "CAD"
    assert names[links[1].mm_index] == "CBD"
    heme = heme_reference_positions(names)
    assert heme is not None
    for link in links:
        ghost = link_atom_position(heme[link.qm_index], heme[link.mm_index])
        assert np.linalg.norm(ghost - heme[link.qm_index]) == pytest.approx(1.09)
        bond = heme[link.mm_index] - heme[link.qm_index]
        assert np.dot(ghost - heme[link.qm_index], bond) > 0.0


def test_link_force_matches_the_chain_rule() -> None:
    r_qm = np.zeros(3)
    r_mm = np.array([1.54, 0.2, -0.1])
    bond = 1.09

    def energy(qm: np.ndarray, mm: np.ndarray) -> float:
        ghost = link_atom_position(qm, mm, bond)
        return float(ghost[1] + 0.3 * ghost[0])

    # dE/dR_ghost = (0.3, 1, 0), so F_ghost = -dE.
    f_ghost = np.array([-0.3, -1.0, 0.0])
    f_qm, f_mm = project_link_force(f_ghost, r_qm, r_mm, bond)
    assert f_qm + f_mm == pytest.approx(f_ghost)
    delta = 1.0e-6
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = delta
        d_qm = (energy(r_qm + step, r_mm) - energy(r_qm - step, r_mm)) / (2 * delta)
        d_mm = (energy(r_qm, r_mm + step) - energy(r_qm, r_mm - step)) / (2 * delta)
        assert f_qm[axis] == pytest.approx(-d_qm, abs=1e-6)
        assert f_mm[axis] == pytest.approx(-d_mm, abs=1e-6)


class _RecordingCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self) -> None:
        super().__init__()
        self.seen = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.seen = atoms.copy()
        n = len(atoms)
        forces = np.zeros((n, 3), dtype=np.float64)
        forces[-1] = np.array([0.0, 1.0, 0.0])
        self.results = {"energy": -2.5, "forces": forces}


def test_whole_system_stamps_charge_and_spin() -> None:
    calc = _RecordingCalculator()
    evaluate_whole_system(
        calc,
        np.array([26, 6], dtype=int),
        np.zeros((2, 3)),
        charge=-2,
        spin_multiplicity=3,
    )
    assert calc.seen is not None
    assert calc.seen.info["charge"] == pytest.approx(-2.0)
    assert calc.seen.info["spin"] == pytest.approx(3.0)
    assert calc.seen.info["spin_multiplicity"] == pytest.approx(3.0)


def test_callback_projects_the_ghost_onto_both_real_atoms() -> None:
    recorder = _RecordingCalculator()
    r_mm = 1.50
    calc = MetatomicMlpotCalculator(
        recorder,
        atomic_numbers=np.array([6, 6], dtype=int),
        atoms_per_monomer=[2],
        eval_mode="whole_system",
        do_mm=False,
        ml_atom_indices=[0],
        charge=-2,
        spin_multiplicity=3,
        link_atoms=(LinkAtom(qm_index=0, mm_index=1, bond_length_A=1.09),),
    )
    x = [0.0, r_mm]
    y = [0.0, 0.0]
    z = [0.0, 0.0]
    dx = [0.0, 0.0]
    dy = [0.0, 0.0]
    dz = [0.0, 0.0]
    energy = calc.calculate_charmm(
        2, 0, 0, None, x, y, z, dx, dy, dz, 0, 0, None, None, None, None, None, None, None
    )
    assert energy == pytest.approx(-2.5 * EV_TO_KCAL_MOL)
    assert recorder.seen is not None
    assert list(recorder.seen.numbers) == [6, 1]
    assert recorder.seen.info["charge"] == pytest.approx(-2.0)
    assert recorder.seen.info["spin"] == pytest.approx(3.0)
    ghost = recorder.seen.positions[1]
    assert np.linalg.norm(ghost - np.zeros(3)) == pytest.approx(1.09)
    g = 1.09 / r_mm
    # Perpendicular ghost force (0, 1, 0) eV/Å splits as (1-g) on QM and g on MM.
    assert dy[0] == pytest.approx(-(1.0 - g) * EV_TO_KCAL_MOL)
    assert dy[1] == pytest.approx(-g * EV_TO_KCAL_MOL)
    assert dx[0] == pytest.approx(0.0)
    assert dx[1] == pytest.approx(0.0)


def test_propionate_model_keeps_tails_out_of_pet(tmp_path) -> None:
    names = _heme_names()
    ckpt = tmp_path / "export.pt"
    ckpt.write_bytes(b"stub")
    args = Namespace(
        residue="HEME",
        n_molecules=1,
        composition=None,
        mm_region="propionates",
        counterions="SOD",
        metatomic_eval_mode="whole_system",
        _cluster_atom_names=[*names, "SOD", "SOD"],
        _cluster_residue_labels=["HEME", "SOD", "SOD"],
    )
    expand_counterions(args)
    model = build_metatomic_mlpot_model(
        ckpt,
        np.array([6] * len(names) + [11, 11], dtype=int),
        [len(names), 1, 1],
        3,
        calculator=_RecordingCalculator(),
        args=args,
        do_mm=False,
        eval_mode="whole_system",
    )
    assert model._charge == 0
    assert model._spin_multiplicity == 3
    assert model._link_atoms is not None
    assert len(model._link_atoms) == 2
    assert model._ml_atom_indices is not None
    assert len(model._ml_atom_indices) == len(names) - len(PROPIONATE_MM_NAMES)
    assert len(names) not in set(model._ml_atom_indices.tolist())
