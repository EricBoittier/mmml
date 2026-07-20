"""Unit tests for mmml.mode_check (no CHARMM / hybrid runtime)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from mmml.cli.__main__ import main as mmml_main
from mmml.cli.misc.mode_check import build_parser
from mmml.cli.registry import command_by_name
from mmml.mode_check import (
    ModeCheckConfig,
    ModeCheckResult,
    force_fd_check,
    reduced_mass_amu,
    run_mode_check,
    spring_constant_to_wavenumber_cm,
)
from mmml.mode_check.bonds import infer_xh_bond_pairs, tip3_oh_pairs
from mmml.mode_check.forces import (
    fit_k_from_force,
    fit_quadratic_k_from_energy,
)
from mmml.mode_check.geometry import build_vacuum_cluster_from_molecules


def test_mode_check_cli_is_registered_and_help_is_reachable(monkeypatch, capsys):
    spec = command_by_name("mode-check")
    assert spec is not None
    assert spec.module == "mmml.cli.misc.mode_check"
    assert build_parser().prog == "mmml mode-check"

    monkeypatch.setattr("sys.argv", ["mmml", "mode-check", "--help"])
    with pytest.raises(SystemExit) as exc:
        mmml_main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "mode-check" in out
    assert "pbc-fd" in out or "--pbc-fd" in out


def test_spring_constant_to_wavenumber_positive():
    # Rough OH scale: k ~ 40 eV/Å² → thousands of cm⁻¹
    nu = spring_constant_to_wavenumber_cm(40.0)
    assert 2000.0 < nu < 5000.0
    assert np.isnan(spring_constant_to_wavenumber_cm(-1.0))


def test_reduced_mass_and_fits():
    mu = reduced_mass_amu(16.0, 1.0)
    assert mu == pytest.approx(16.0 / 17.0)
    deltas = np.linspace(-0.03, 0.03, 7)
    k_true = 25.0
    energies = 0.5 * k_true * deltas**2
    forces = -k_true * deltas
    assert fit_quadratic_k_from_energy(deltas, energies) == pytest.approx(k_true, rel=1e-6)
    assert fit_k_from_force(deltas, forces) == pytest.approx(k_true, rel=1e-6)


def test_tip3_oh_pairs_and_infer():
    assert tip3_oh_pairs(1) == [(0, 1), (0, 2)]
    assert tip3_oh_pairs(2) == [(0, 1), (0, 2), (3, 4), (3, 5)]
    atoms, apm, labels = build_vacuum_cluster_from_molecules([("TIP3", 2)], separation_A=2.8)
    assert labels == ["TIP3", "TIP3"]
    assert apm == [3, 3]
    pairs = infer_xh_bond_pairs(
        atoms.get_atomic_numbers(),
        atoms.get_positions(),
        atoms_per_monomer=apm,
    )
    assert len(pairs) == 4
    assert all(atoms.get_atomic_numbers()[i] == 8 for i, _ in pairs)
    assert all(atoms.get_atomic_numbers()[j] == 1 for _, j in pairs)


def test_run_mode_check_fd_and_bond_scan_on_harmonic(tmp_path: Path):
    # Two-atom "OH" with analytic harmonic potential
    from ase.calculators.calculator import Calculator

    class HarmonicOH(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, k=30.0, r0=0.96, **kwargs):
            super().__init__(**kwargs)
            self.k = float(k)
            self.r0 = float(r0)

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            pos = atoms.get_positions()
            vec = pos[1] - pos[0]
            r = float(np.linalg.norm(vec))
            u = vec / r
            e = 0.5 * self.k * (r - self.r0) ** 2
            f_mag = -self.k * (r - self.r0)
            forces = np.zeros_like(pos)
            forces[0] = -f_mag * u
            forces[1] = f_mag * u
            self.results = {"energy": e, "forces": forces}

    atoms = Atoms(
        numbers=[8, 1],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
    )
    atoms.calc = HarmonicOH(k=30.0, r0=0.96)
    cfg = ModeCheckConfig(
        checks=("fd", "bond-scan"),
        fd_atoms=2,
        fd_dx_A=1e-4,
        atoms_per_monomer=(2,),
    )
    result = run_mode_check(atoms, cfg, output_dir=tmp_path)
    assert isinstance(result, ModeCheckResult)
    assert result.fd is not None
    assert result.fd["fd_force_max_abs_diff_eVA"] < 1e-3
    assert "XH0" in result.bond_scans
    assert result.bond_scans["XH0"]["k_from_E_eV_A2"] == pytest.approx(30.0, rel=5e-2)
    assert (tmp_path / "mode_check_summary.json").is_file()


def test_force_fd_check_with_emt():
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.calc = EMT()
    out = force_fd_check(atoms, natoms_check=2, dx=1e-4)
    assert out["fd_atoms_checked"] == 2.0
    assert out["fd_force_max_abs_diff_eVA"] < 0.05


def test_hybrid_setup_disables_mm_for_monomer():
    from mmml.mode_check import HybridModeCheckSetup

    setup = HybridModeCheckSetup(
        composition=(("TIP3", 1),),
        checkpoint=Path("/tmp/fake.json"),
        do_mm=True,
    )
    assert setup.do_mm is False
    assert setup.do_ml_dimer is False


def test_place_monomers_along_x_and_reject_collapsed_geometry():
    from mmml.mode_check.hybrid import (
        assert_resolved_vacuum_geometry,
        com_separations_along_chain,
        place_monomers_along_x,
    )

    tip3 = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        dtype=float,
    )
    geoms = {"TIP3": (tip3, ["OH2", "H1", "H2"], np.array([8, 1, 1]))}
    placed = place_monomers_along_x(
        geoms, ["TIP3", "TIP3"], [3, 3], separation_A=2.8
    )
    assert placed.shape == (6, 3)
    assert np.linalg.norm(placed[1] - placed[0]) == pytest.approx(0.96, abs=1e-6)
    # Second monomer COM near x=2.8
    assert placed[3:6].mean(axis=0)[0] == pytest.approx(2.8, abs=1e-6)
    assert com_separations_along_chain(placed, [3, 3]) == pytest.approx([2.8], abs=1e-6)
    assert_resolved_vacuum_geometry(placed, [3, 3])

    with pytest.raises(RuntimeError, match="collapsed|coincident|intramolecular"):
        assert_resolved_vacuum_geometry(np.zeros((3, 3)), [3])


def test_mode_check_far_vs_separation_cli():
    from mmml.cli.misc.mode_check import build_parser, _resolve_monomer_separation_A
    from mmml.mode_check.config import (
        DEFAULT_MONOMER_SEPARATION_A,
        FAR_MONOMER_SEPARATION_A,
    )

    p = build_parser()
    assert (
        _resolve_monomer_separation_A(p.parse_args([]), n_monomers=1)
        == DEFAULT_MONOMER_SEPARATION_A
    )
    # Dimers default far (unoriented 2.8 Å COM often clashes).
    assert (
        _resolve_monomer_separation_A(p.parse_args([]), n_monomers=2)
        == FAR_MONOMER_SEPARATION_A
    )
    assert (
        _resolve_monomer_separation_A(p.parse_args(["--far"]), n_monomers=2)
        == FAR_MONOMER_SEPARATION_A
    )
    assert _resolve_monomer_separation_A(
        p.parse_args(["--monomer-separation", "12.5"]), n_monomers=2
    ) == pytest.approx(12.5)
    with pytest.raises(SystemExit):
        _resolve_monomer_separation_A(
            p.parse_args(["--far", "--monomer-separation", "3.0"]),
            n_monomers=2,
        )


def test_fix_monomer_coms_preserves_com_under_forces():
    from ase import Atoms

    from mmml.mode_check.constraints import FixMonomerCOMs

    atoms = Atoms(
        numbers=[8, 1, 1, 8, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [6.0, 0.0, 0.0],
            [6.96, 0.0, 0.0],
            [5.76, 0.93, 0.0],
        ],
        masses=[16.0, 1.0, 1.0, 16.0, 1.0, 1.0],
    )
    # Place COMs exactly on the x-axis spacing used below.
    cons = FixMonomerCOMs(atoms, [3, 3])
    com0 = atoms.get_center_of_mass(indices=[0, 1, 2])
    com1 = atoms.get_center_of_mass(indices=[3, 4, 5])
    forces = np.zeros((6, 3))
    forces[0] = [1.0, 0.0, 0.0]
    forces[3] = [-2.0, 0.0, 0.0]
    cons.adjust_forces(atoms, forces)
    # Net force on each monomer must vanish (mass-weighted).
    m = atoms.get_masses()
    assert (m[:3] @ forces[:3]) == pytest.approx(0.0, abs=1e-12)
    assert (m[3:] @ forces[3:]) == pytest.approx(0.0, abs=1e-12)
    new = atoms.get_positions() + np.array([[0.1, 0, 0]] * 6)
    cons.adjust_positions(atoms, new)
    atoms2 = atoms.copy()
    atoms2.set_positions(new)
    assert atoms2.get_center_of_mass(indices=[0, 1, 2]) == pytest.approx(com0, abs=1e-9)
    assert atoms2.get_center_of_mass(indices=[3, 4, 5]) == pytest.approx(com1, abs=1e-9)


def test_minimize_with_frozen_coms_keeps_separation():
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from mmml.mode_check import ModeCheckConfig, run_mode_check

    class SoftSpring(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            Calculator.calculate(self, atoms, properties, system_changes)
            pos = atoms.get_positions()
            # Pull every atom toward origin (would collapse COM without constraint).
            forces = -0.5 * pos
            energy = 0.25 * float(np.sum(pos**2))
            self.results = {"energy": energy, "forces": forces}

    atoms = Atoms(
        numbers=[8, 1, 8, 1],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        masses=[16.0, 1.0, 16.0, 1.0],
    )
    atoms.calc = SoftSpring()
    com_sep0 = float(
        np.linalg.norm(
            atoms.get_center_of_mass(indices=[0, 1])
            - atoms.get_center_of_mass(indices=[2, 3])
        )
    )
    cfg = ModeCheckConfig(
        checks=("minimize",),
        minimize_fmax=0.05,
        minimize_steps=50,
        minimize_freeze_monomer_coms=True,
        atoms_per_monomer=(2, 2),
    )
    result = run_mode_check(atoms, cfg)
    assert "minimize" not in result.errors
    com_sep1 = float(
        np.linalg.norm(
            atoms.get_center_of_mass(indices=[0, 1])
            - atoms.get_center_of_mass(indices=[2, 3])
        )
    )
    assert com_sep1 == pytest.approx(com_sep0, abs=1e-6)


def test_cutoff_region_stations_cover_handoff_ruler():
    from mmml.mode_check.cutoff_ladder import cutoff_region_stations, region_boundaries

    stations = cutoff_region_stations(
        ml_switch_width=1.5, mm_switch_on=6.0, mm_switch_width=5.0
    )
    labels = [s.label for s in stations]
    assert labels == [
        "ml_interior",
        "ml_edge",
        "handoff_mid",
        "mm_switch_on",
        "mm_tail_mid",
        "mm_off",
        "beyond",
    ]
    by_label = {s.label: s.com_A for s in stations}
    assert by_label["ml_edge"] == pytest.approx(4.5)
    assert by_label["handoff_mid"] == pytest.approx(5.25)
    assert by_label["mm_switch_on"] == pytest.approx(6.0)
    assert by_label["mm_tail_mid"] == pytest.approx(8.5)
    assert by_label["mm_off"] == pytest.approx(11.0)
    assert by_label["beyond"] == pytest.approx(15.0)
    # Strictly increasing
    coms = [s.com_A for s in stations]
    assert coms == sorted(coms)
    bounds = region_boundaries(
        ml_switch_width=1.5, mm_switch_on=6.0, mm_switch_width=5.0
    )
    assert bounds["r_ml_edge_A"] == pytest.approx(4.5)
    assert bounds["r_mm_off_A"] == pytest.approx(11.0)


def test_reposition_monomers_preserves_internal_geometry():
    from ase import Atoms

    from mmml.mode_check.hybrid import (
        com_separations_along_chain,
        reposition_monomers_along_x,
    )

    atoms = Atoms(
        numbers=[8, 1, 1, 8, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [10.0, 0.0, 0.0],
            [10.96, 0.0, 0.0],
            [9.76, 0.93, 0.0],
        ],
    )
    r_oh0 = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    reposition_monomers_along_x(atoms, [3, 3], separation_A=6.0)
    assert com_separations_along_chain(atoms.positions, [3, 3]) == pytest.approx(
        [6.0], abs=1e-6
    )
    assert float(np.linalg.norm(atoms.positions[1] - atoms.positions[0])) == pytest.approx(
        r_oh0, abs=1e-6
    )


def test_intermolecular_clash_rejected_at_close_com():
    from mmml.mode_check.hybrid import (
        assert_resolved_vacuum_geometry,
        min_intermolecular_distance_A,
        place_monomers_along_x,
    )

    # Elongated "water" so COM=2.8 Å still has atom clashes.
    tip3 = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]],
        dtype=float,
    )
    geoms = {"TIP3": (tip3, ["OH2", "H1", "H2"], np.array([8, 1, 1]))}
    placed = place_monomers_along_x(
        geoms, ["TIP3", "TIP3"], [3, 3], separation_A=2.8
    )
    d_ij = min_intermolecular_distance_A(placed, [3, 3])
    assert d_ij is not None and d_ij < 1.2
    with pytest.raises(RuntimeError, match="inter-monomer"):
        assert_resolved_vacuum_geometry(placed, [3, 3])
    # Far placement is fine.
    far = place_monomers_along_x(
        geoms, ["TIP3", "TIP3"], [3, 3], separation_A=15.0
    )
    assert_resolved_vacuum_geometry(far, [3, 3])
