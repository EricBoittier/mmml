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
    ORIENTATION_HBOND,
    SCHEMA_VERSION,
    SYSTEM_WATER,
    dimer_at_distance,
    dump_interaction_pes_json,
    equilateral_trimer,
    linspace_angstrom,
    load_interaction_pes_json,
    load_monomer_xyz,
    orient_monomer_pair,
    run_interaction_pes_campaign,
    scan_dimer_slice,
    trimer_mbe_ev,
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


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.r_min == pytest.approx(DEFAULT_R_MIN_A)
    assert args.r_max == pytest.approx(DEFAULT_R_MAX_A)
    assert args.n_r == 20
    assert args.n_r_2d == 12
    assert args.n_theta == 12
    assert args.surface_system == "ethanol"
    assert args.include_acetone is True
    assert args.from_json is None
    help_text = build_parser().format_help()
    assert "examples/orca/water_opt/water.xyz" in help_text
    assert str(REPO.resolve()) not in help_text


def test_cli_missing_checkpoint(tmp_path: Path) -> None:
    rc = main(["--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["ok"] is False
    assert "missing.pt" in report["checkpoint"]


def test_dimer_com_distance_matches_request() -> None:
    monomer = _water()
    a, b = orient_monomer_pair(monomer, ORIENTATION_HBOND)
    dimer = dimer_at_distance(a, b, 5.0)
    n = len(a)
    com_a = dimer[:n].get_center_of_mass()
    com_b = dimer[n:].get_center_of_mass()
    assert float(np.linalg.norm(com_b - com_a)) == pytest.approx(5.0, abs=1e-6)


def test_equilateral_trimer_side_length() -> None:
    trimer = equilateral_trimer(_water(), 4.2)
    ids = np.asarray(trimer.arrays["mol_id"])
    coms = [
        trimer[ids == mol].get_center_of_mass()
        for mol in (0, 1, 2)
    ]
    sides = [
        float(np.linalg.norm(coms[0] - coms[1])),
        float(np.linalg.norm(coms[0] - coms[2])),
        float(np.linalg.norm(coms[1] - coms[2])),
    ]
    assert sides[0] == pytest.approx(4.2, abs=1e-6)
    assert sides[1] == pytest.approx(4.2, abs=1e-6)
    assert sides[2] == pytest.approx(4.2, abs=1e-6)


def test_pair_calculator_has_vanishing_e3() -> None:
    trimer = equilateral_trimer(_water(), 3.0)
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
    monomer = _water()
    a, b = orient_monomer_pair(monomer, ORIENTATION_HBOND)
    dimer = dimer_at_distance(a, b, 4.0)
    cache: dict[str, float] = {}
    from mmml.analysis.interaction_pes import interaction_energy_ev

    e_int, e_ab, monomers = interaction_energy_ev(dimer, FragmentPairCalculator, cache)
    assert e_int == pytest.approx(e_ab - monomers[0] - monomers[1], abs=1e-12)
    assert e_int == pytest.approx(1.0 / 4.0, rel=1e-6)


def test_campaign_json_schema_and_replot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monomer = centered_atoms(Atoms("OH2", positions=[[0, 0, 0.1], [0.8, 0, -0.4], [-0.8, 0, -0.4]]), center="com")
    document = run_interaction_pes_campaign(
        calculator_factory=FragmentThreeBodyCalculator,
        systems={SYSTEM_WATER: monomer},
        r_1d=linspace_angstrom(3.0, 12.0, 4),
        r_2d=linspace_angstrom(3.5, 6.0, 3),
        theta_deg=linspace_angstrom(0.0, 90.0, 3),
        r_trimer=linspace_angstrom(3.0, 12.0, 4),
        slice_systems=(SYSTEM_WATER,),
        surface_system=SYSTEM_WATER,
        trimer_systems=(SYSTEM_WATER,),
        calculator_name="dummy",
    )
    assert document["schema"] == SCHEMA_VERSION
    assert document["energy_definition"] == "interaction"
    assert document["units"]["energy"] == "kcal/mol"
    assert document["units"]["distance"] == "angstrom"
    assert len(document["dimer_slices"]) == 2
    assert document["dimer_surfaces"][0]["system"] == SYSTEM_WATER
    z = np.asarray(document["dimer_surfaces"][0]["e_int_kcal_mol"])
    assert z.shape == (3, 3)
    tri = document["trimer_slices"][0]
    assert "e3_kcal_mol" in tri
    assert abs(tri["e3_kcal_mol"][0]) > abs(tri["e3_kcal_mol"][-1])
    path = dump_interaction_pes_json(document, tmp_path / "interaction_pes.json")
    loaded = load_interaction_pes_json(path)
    assert loaded["summary"]["water_hbond_far_field_kcal_mol"] == pytest.approx(
        document["dimer_slices"][0]["far_field_kcal_mol"]
    )
    from mmml.analysis.interaction_pes_plot import write_interaction_pes_figures

    figures = write_interaction_pes_figures(loaded, tmp_path / "fig", prefix="dummy")
    assert figures["slices"].is_file()
    assert figures["surface"].is_file()
    assert figures["trimer"].is_file()
    assert figures["slices"].with_suffix(".pdf").is_file()


def test_cli_from_json_replots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monomer = _water()
    document = run_interaction_pes_campaign(
        calculator_factory=FragmentPairCalculator,
        systems={SYSTEM_WATER: monomer},
        r_1d=[3.0, 6.0, 12.0],
        r_2d=[4.0, 5.0],
        theta_deg=[0.0, 90.0],
        r_trimer=[3.0, 12.0],
        slice_systems=(SYSTEM_WATER,),
        surface_system=SYSTEM_WATER,
        trimer_systems=(SYSTEM_WATER,),
        calculator_name="dummy",
    )
    json_path = dump_interaction_pes_json(document, tmp_path / "in.json")
    out = tmp_path / "plots"
    rc = main(["--from-json", str(json_path), "--output-dir", str(out), "--prefix", "unit"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["ok"] is True
    assert Path(report["figures"]["slices"]).is_file()


def test_kcal_conversion_matches_units_module() -> None:
    monomer = _water()
    row = scan_dimer_slice(
        monomer,
        [5.0],
        orientation=ORIENTATION_HBOND,
        calculator_factory=FragmentPairCalculator,
        cache={},
        system=SYSTEM_WATER,
    )
    assert row["e_int_kcal_mol"][0] == pytest.approx(row["e_int_ev"][0] * EV_TO_KCAL_MOL)


def test_cli_is_registered() -> None:
    from mmml.cli.registry import command_by_name

    spec = command_by_name("pet-interaction-pes")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.pet_interaction_pes"
    assert spec.summary
