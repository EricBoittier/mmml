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
