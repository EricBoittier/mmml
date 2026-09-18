"""Interaction PES geometry, MBE algebra, JSON schema, and CLI (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mmml.analysis.dimer_scans import centered_atoms
from mmml.analysis.interaction_pes import (
    DEFAULT_R_MAX_A,
    DEFAULT_R_MIN_A,
    ORIENTATION_ACCEPTOR_ACCEPTOR,
    ORIENTATION_LINEAR_OH_O,
    SCHEMA_VERSION,
    SYSTEM_WATER,
    dump_interaction_pes_json,
    dump_interaction_pes_npz,
    equilateral_trimer,
    linspace_angstrom,
    load_interaction_pes_json,
    load_monomer_xyz,
    run_interaction_pes_campaign,
    scan_dimer_slice,
    trimer_mbe_ev,
)
from mmml.analysis.interaction_pes_geom import (
    MOTIF_CYCLIC,
    MOTIF_LINEAR,
    cyclic_hbond_trimer,
    dimer_acceptor_acceptor,
    dimer_dha_deg,
    dimer_oh_o,
    dimer_site_indices,
    linear_hbond_trimer,
    polar_angle_for_dha,
    site_site_distance,
    trimer_oo_distances,
)
from mmml.cli.misc.pet_interaction_pes import build_parser, main
from mmml.data.units import EV_TO_KCAL_MOL

REPO = Path(__file__).resolve().parents[2]
WATER_XYZ = REPO / "examples" / "orca" / "water_opt" / "water.xyz"
PLOT_MODULE = REPO / "mmml" / "analysis" / "interaction_pes_plot.py"


class FragmentPairCalculator(Calculator):
    """``E = N + sum_{I<J} 1/r_COM`` — pairwise in the molecular fragments."""

    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        energy = float(len(atoms))
        if "mol_id" in atoms.arrays:
            mol_id = np.asarray(atoms.arrays["mol_id"])
            ids = np.unique(mol_id)
            coms = []
            masses = np.asarray(atoms.get_masses(), dtype=np.float64)
            pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            for mol in ids:
                mask = mol_id == mol
                coms.append(np.average(pos[mask], axis=0, weights=masses[mask]))
            for i in range(len(coms)):
                for j in range(i + 1, len(coms)):
                    energy += 1.0 / float(np.linalg.norm(coms[i] - coms[j]))
        self.results["energy"] = energy


class FragmentThreeBodyCalculator(FragmentPairCalculator):
    """Pair terms plus ``1/(r_AB r_AC r_BC)`` on trimers."""

    three_body_scale = 8.0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        if "mol_id" not in atoms.arrays:
            return
        mol_id = np.asarray(atoms.arrays["mol_id"])
        ids = np.unique(mol_id)
        if ids.size != 3:
            return
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        coms = [
            np.average(pos[mol_id == mol], axis=0, weights=masses[mol_id == mol])
            for mol in ids
        ]
        r_ab = float(np.linalg.norm(coms[0] - coms[1]))
        r_ac = float(np.linalg.norm(coms[0] - coms[2]))
        r_bc = float(np.linalg.norm(coms[1] - coms[2]))
        self.results["energy"] = float(self.results["energy"]) + self.three_body_scale / (
            r_ab * r_ac * r_bc
        )


def _water() -> Atoms:
    return load_monomer_xyz(WATER_XYZ)


def test_plot_module_uses_shared_icml_style() -> None:
    source = PLOT_MODULE.read_text(encoding="utf-8")
    assert "from mmml.utils.plotting.styles import apply_plot_style" in source
    assert 'apply_plot_style("icml")' in source
    assert "comparison_colors" in source


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.r_min == pytest.approx(DEFAULT_R_MIN_A)
    assert args.r_max == pytest.approx(DEFAULT_R_MAX_A)
    assert args.n_r == 0
    assert args.n_r_2d == 0
    assert args.n_theta == 13
    assert args.theta_min == pytest.approx(120.0)
    assert args.surface_system == "ethanol"
    assert args.include_acetone is True
    assert args.from_json is None
    help_text = build_parser().format_help()
    assert "examples/orca/water_opt/water.xyz" in help_text
    assert str(REPO.resolve()) not in help_text
    assert "piecewise" in help_text


def test_cli_missing_checkpoint(tmp_path: Path) -> None:
    rc = main(["--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["ok"] is False
    assert "missing.pt" in report["checkpoint"]


def test_linear_oh_o_is_linear_and_matches_oo() -> None:
    monomer = _water()
    dimer = dimer_oh_o(monomer, 2.90, dha_deg=180.0)
    i, j = dimer_site_indices(dimer, ORIENTATION_LINEAR_OH_O)
    assert site_site_distance(dimer, i, j) == pytest.approx(2.90, abs=1e-6)
    assert dimer_dha_deg(dimer) == pytest.approx(180.0, abs=0.5)
    # Cs acceptor flap (57°) leaves O–O and DHA unchanged and lifts acceptor H's off y=0.
    acc = dimer[len(monomer) :]
    assert float(np.max(np.abs(acc.get_positions()[:, 1]))) > 0.2


def test_bent_oh_o_recovers_requested_dha() -> None:
    dimer = dimer_oh_o(_water(), 2.90, dha_deg=150.0, plane="xz")
    assert dimer_dha_deg(dimer) == pytest.approx(150.0, abs=0.8)
    dimer_yz = dimer_oh_o(_water(), 2.90, dha_deg=140.0, plane="yz")
    assert dimer_dha_deg(dimer_yz) == pytest.approx(140.0, abs=0.8)
    linear_xz = dimer_oh_o(_water(), 2.90, dha_deg=180.0, plane="xz")
    linear_yz = dimer_oh_o(_water(), 2.90, dha_deg=180.0, plane="yz")
    assert np.allclose(linear_xz.get_positions(), linear_yz.get_positions(), atol=1e-6)


def test_polar_angle_limits() -> None:
    assert polar_angle_for_dha(2.90, 0.96, 180.0) == pytest.approx(0.0, abs=1e-9)
    gamma90 = polar_angle_for_dha(2.90, 0.96, 90.0)
    assert gamma90 == pytest.approx(float(np.arccos(0.96 / 2.90)), rel=1e-5)


def test_acceptor_acceptor_oo_distance() -> None:
    dimer = dimer_acceptor_acceptor(_water(), 3.20)
    i, j = dimer_site_indices(dimer, ORIENTATION_ACCEPTOR_ACCEPTOR)
    assert site_site_distance(dimer, i, j) == pytest.approx(3.20, abs=1e-6)


def test_cyclic_and_linear_trimer_oo() -> None:
    cyclic = cyclic_hbond_trimer(_water(), 2.85)
    sides = trimer_oo_distances(cyclic)
    assert sides[0] == pytest.approx(2.85, abs=1e-5)
    assert sides[1] == pytest.approx(2.85, abs=1e-5)
    assert sides[2] == pytest.approx(2.85, abs=1e-5)
    linear = linear_hbond_trimer(_water(), 2.85)
    ab, ac, bc = trimer_oo_distances(linear)
    assert ab == pytest.approx(2.85, abs=1e-5)
    assert bc == pytest.approx(2.85, abs=1e-5)
    assert ac == pytest.approx(5.70, abs=1e-4)


def test_equilateral_trimer_side_length() -> None:
    trimer = equilateral_trimer(_water(), 4.2)
    ids = np.asarray(trimer.arrays["mol_id"])
    coms = [trimer[ids == mol].get_center_of_mass() for mol in (0, 1, 2)]
    sides = [
        float(np.linalg.norm(coms[0] - coms[1])),
        float(np.linalg.norm(coms[0] - coms[2])),
        float(np.linalg.norm(coms[1] - coms[2])),
    ]
    assert sides[0] == pytest.approx(4.2, abs=1e-6)
    assert sides[1] == pytest.approx(4.2, abs=1e-6)
    assert sides[2] == pytest.approx(4.2, abs=1e-6)


def test_pair_calculator_has_vanishing_e3() -> None:
    trimer = cyclic_hbond_trimer(_water(), 3.0)
    mbe = trimer_mbe_ev(trimer, FragmentPairCalculator, cache={})
    assert mbe["e3_ev"] == pytest.approx(0.0, abs=1e-9)
    assert mbe["e_int_ev"] == pytest.approx(mbe["e_pair_sum_ev"], abs=1e-9)


def test_three_body_calculator_has_nonzero_e3() -> None:
    trimer = equilateral_trimer(_water(), 3.0)
    mbe = trimer_mbe_ev(trimer, FragmentThreeBodyCalculator, cache={})
    expected = FragmentThreeBodyCalculator.three_body_scale / (3.0**3)
    assert mbe["e3_ev"] == pytest.approx(expected, rel=1e-6)
    assert abs(mbe["e3_ev"]) > 0.05


def test_interaction_energy_is_dimer_minus_monomers() -> None:
    dimer = dimer_oh_o(_water(), 4.0)
    cache: dict[str, float] = {}
    from mmml.analysis.interaction_pes import interaction_energy_ev

    e_int, e_ab, monomers = interaction_energy_ev(dimer, FragmentPairCalculator, cache)
    assert e_int == pytest.approx(e_ab - monomers[0] - monomers[1], abs=1e-12)
    n = 3
    com_a = dimer[:n].get_center_of_mass()
    com_b = dimer[n:].get_center_of_mass()
    r_com = float(np.linalg.norm(com_b - com_a))
    assert e_int == pytest.approx(1.0 / r_com, rel=1e-6)


def test_campaign_json_schema_and_replot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monomer = centered_atoms(
        Atoms("OH2", positions=[[0, 0, 0.1], [0.8, 0, -0.4], [-0.8, 0, -0.4]]),
        center="com",
    )
    document = run_interaction_pes_campaign(
        calculator_factory=FragmentThreeBodyCalculator,
        systems={SYSTEM_WATER: monomer},
        r_1d=linspace_angstrom(3.0, 12.0, 4),
        r_2d=linspace_angstrom(3.5, 6.0, 3),
        theta_deg=linspace_angstrom(140.0, 180.0, 3),
        r_trimer=linspace_angstrom(3.0, 12.0, 4),
        slice_systems=(SYSTEM_WATER,),
        surface_system=SYSTEM_WATER,
        angular_systems=(SYSTEM_WATER,),
        trimer_systems=(SYSTEM_WATER,),
        trimer_motifs={SYSTEM_WATER: (MOTIF_CYCLIC, MOTIF_LINEAR)},
        calculator_name="dummy",
    )
    assert document["schema"] == SCHEMA_VERSION
    assert document["energy_definition"] == "interaction"
    assert document["units"]["energy"] == "kcal/mol"
    assert document["units"]["distance"] == "angstrom"
    assert len(document["dimer_slices"]) == 2
    assert {row["orientation"] for row in document["dimer_slices"]} == {
        ORIENTATION_LINEAR_OH_O,
        ORIENTATION_ACCEPTOR_ACCEPTOR,
    }
    assert document["dimer_slices"][0]["scan_coordinate"] == "O-O"
    assert document["dimer_surfaces"][0]["system"] == SYSTEM_WATER
    assert document["dimer_surfaces"][0]["angle_name"] == "donor_h_acceptor"
    z = np.asarray(document["dimer_surfaces"][0]["e_int_kcal_mol"])
    assert z.shape == (3, 3)
    assert len(document["dimer_angular"]) == 2
    tri = document["trimer_slices"][0]
    assert "e3_kcal_mol" in tri
    assert "e3_over_eint" in tri
    assert tri["motif"] in {MOTIF_CYCLIC, MOTIF_LINEAR}
    path = dump_interaction_pes_json(document, tmp_path / "interaction_pes.json")
    npz = dump_interaction_pes_npz(document, tmp_path / "interaction_pes.npz")
    loaded = load_interaction_pes_json(path)
    assert loaded["summary"]["water_linear_oh_o_far_field_kcal_mol"] == pytest.approx(
        document["dimer_slices"][0]["far_field_kcal_mol"]
    )
    assert npz.is_file()
    from mmml.analysis.interaction_pes_plot import write_interaction_pes_figures

    figures = write_interaction_pes_figures(loaded, tmp_path / "fig", prefix="dummy")
    assert figures["slices"].is_file()
    assert figures["surface"].is_file()
    assert figures["trimer"].is_file()
    assert figures["angular"].is_file()
    assert figures["slices"].with_suffix(".pdf").is_file()


def test_cli_from_json_replots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monomer = _water()
    document = run_interaction_pes_campaign(
        calculator_factory=FragmentPairCalculator,
        systems={SYSTEM_WATER: monomer},
        r_1d=[3.0, 6.0, 12.0],
        r_2d=[4.0, 5.0],
        theta_deg=[150.0, 180.0],
        r_trimer=[3.0, 12.0],
        slice_systems=(SYSTEM_WATER,),
        surface_system=SYSTEM_WATER,
        angular_systems=(SYSTEM_WATER,),
        trimer_systems=(SYSTEM_WATER,),
        trimer_motifs={SYSTEM_WATER: (MOTIF_CYCLIC,)},
        calculator_name="dummy",
    )
    json_path = dump_interaction_pes_json(document, tmp_path / "in.json")
    out = tmp_path / "plots"
    rc = main(["--from-json", str(json_path), "--output-dir", str(out), "--prefix", "unit"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["ok"] is True
    assert Path(report["figures"]["slices"]).is_file()
    assert Path(report["figures"]["angular"]).is_file()


def test_mask_clash_energy_nans_short_contacts() -> None:
    from mmml.analysis.interaction_pes import mask_clash_energy

    energy = np.array([-3.0, 1e4, 0.1])
    contact = np.array([2.2, 0.3, 2.5])
    masked = mask_clash_energy(energy, contact, min_contact_A=2.0)
    assert masked[0] == pytest.approx(-3.0)
    assert np.isnan(masked[1])
    assert masked[2] == pytest.approx(0.1)
    row = scan_dimer_slice(
        _water(),
        [5.0],
        orientation=ORIENTATION_LINEAR_OH_O,
        calculator_factory=FragmentPairCalculator,
        cache={},
        system=SYSTEM_WATER,
    )
    assert row["e_int_kcal_mol"][0] == pytest.approx(row["e_int_ev"][0] * EV_TO_KCAL_MOL)
    assert row["scan_coordinate"] == "O-O"


def test_committed_campaign_json_schema() -> None:
    path = REPO / "examples" / "pet_mad_etoh_pbc" / "data" / "interaction_pes.json"
    document = load_interaction_pes_json(path)
    assert document["schema"] == SCHEMA_VERSION
    assert document["energy_definition"] == "interaction"
    assert document["units"]["energy"] == "kcal/mol"
    systems = {row["system"] for row in document["dimer_slices"]}
    assert {"water", "ethanol", "acetone"} <= systems
    orientations = {row["orientation"] for row in document["dimer_slices"]}
    assert ORIENTATION_LINEAR_OH_O in orientations
    assert document["summary"]["water_linear_oh_o_far_field_kcal_mol"] == pytest.approx(0.0, abs=1e-3)
    assert document["summary"]["ethanol_linear_oh_o_far_field_kcal_mol"] == pytest.approx(0.0, abs=1e-3)
    cyclic_keys = [k for k in document["summary"] if k.endswith("cyclic_trimer_e3_peak_kcal_mol")]
    assert cyclic_keys
    assert any(abs(document["summary"][k]) > 0.5 for k in cyclic_keys)
    assert document["dimer_surfaces"][0]["angle_name"] == "donor_h_acceptor"
    assert document.get("dimer_angular")


def test_cli_is_registered() -> None:
    from mmml.cli.registry import command_by_name

    spec = command_by_name("pet-interaction-pes")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.pet_interaction_pes"
    assert spec.summary
