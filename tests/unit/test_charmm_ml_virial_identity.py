"""Atomic virial vs strain -dE/dV (CHARMM CPT vs JAX-MD #249).

``P_atomic = (Σ F·r) / 3V`` is what CHARMM ``VIRAL`` / ``PRSI`` integrates.
``P_strain = -dE/dV`` at fixed fractional coordinates is what JAX-MD NPT uses.

They agree for an interior pair (MIC = Cartesian). They disagree for a
cross-boundary MIC pair when the lattice shift is stop-grad — the H2 detector
for a hybrid PBC walk that CPT would follow and JAX-MD would not.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.virial_compare import (
    classify_cpt_ml_virial_hypothesis,
    collect_toy_mic_pressures,
    mic_harmonic_energy_ev,
    mic_harmonic_forces_stopgrad_ev,
    pressures_agree,
    strain_pressure_atm,
    virial_pressure_atm,
    wrap_delta,
)


def test_atomic_virial_matches_validate_script_ideal_gas():
    """Same identity ``scripts/validate_virial_vs_charmm.py`` pins (no CHARMM)."""
    from scripts.validate_virial_vs_charmm import (
        virial_pressure_atm as script_virial,
    )

    n, volume, ke = 100, 1000.0, 5.0
    zeros = np.zeros((n, 3))
    assert virial_pressure_atm(zeros, zeros, volume, kinetic_ev=ke) == pytest.approx(
        script_virial(zeros, zeros, volume, kinetic_ev=ke), rel=1e-15
    )


def test_interior_mic_pair_atomic_matches_strain():
    """No wrap: Euler's theorem, P_atomic ≈ P_strain."""
    box = 10.0
    frac = np.array([[0.40, 0.50, 0.50], [0.50, 0.50, 0.50]])
    out = collect_toy_mic_pressures(frac, box, k=2.0, rel_dv=1e-5)
    pos = frac * box
    assert np.max(np.abs(wrap_delta(pos[1] - pos[0], box) - (pos[1] - pos[0]))) < 1e-12
    assert out["p_atomic_atm"] == pytest.approx(out["p_strain_atm"], rel=1e-4)
    assert abs(out["p_atomic_atm"]) > 1e-6


def test_cross_boundary_mic_pair_is_h2_detector():
    """Wrapped pair: CHARMM-style Σ F·r disagrees with -dE/dV (known factor 4)."""
    box = 10.0
    k = 1.0
    frac = np.array([[0.10, 0.50, 0.50], [0.90, 0.50, 0.50]])
    out = collect_toy_mic_pressures(frac, box, k=k, rel_dv=1e-5)
    pos = frac * box
    d_mic = wrap_delta(pos[1] - pos[0], box)
    assert d_mic[0] == pytest.approx(-2.0)
    # |d| = 0.2 L → E = 0.02 k L^2; dE/dV = 0.04 k / (3L); |P_atomic|/|P_strain| = 4
    assert out["p_atomic_atm"] / out["p_strain_atm"] == pytest.approx(-4.0, rel=1e-3)
    assert not pressures_agree(out["p_atomic_atm"], out["p_strain_atm"], rel=0.15, abs_atm=50.0)
    label, _ = classify_cpt_ml_virial_hypothesis(
        p_prsi_atm=out["p_atomic_atm"],
        p_atomic_atm=out["p_atomic_atm"],
        p_strain_atm=out["p_strain_atm"],
    )
    assert label == "H2"


def test_strain_fd_matches_analytic_interior_harmonic():
    """Independent check: dE/dV of E = ½ k (s L)^2 at fixed s."""
    box = 12.0
    k = 3.0
    frac = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    p_fd = strain_pressure_atm(
        lambda r, side: mic_harmonic_energy_ev(r, side, k=k),
        frac,
        box,
        rel_dv=1e-6,
    )
    # |d| = 0.15 L, E = 0.5 k (0.15 L)^2, dE/dV = 0.0225 k / (3L)
    p_analytic = -(0.0225 * k / (3.0 * box)) * (
        1.602176634e-19 / 1e-30 / 101325.0
    )
    assert p_fd == pytest.approx(p_analytic, rel=1e-5)


def test_stopgrad_forces_match_finite_difference_at_fixed_l():
    box = 10.0
    pos = np.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
    analytic = mic_harmonic_forces_stopgrad_ev(pos, box, k=1.5)
    h = 1e-6
    fd = np.zeros_like(pos)
    e0 = mic_harmonic_energy_ev(pos, box, k=1.5)
    for i in range(2):
        for c in range(3):
            pert = pos.copy()
            pert[i, c] += h
            fd[i, c] = -(mic_harmonic_energy_ev(pert, box, k=1.5) - e0) / h
    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)


def test_live_diagnose_argv_parses(tmp_path):
    """md-system accepts the functionality-script argv (no CHARMM)."""
    from mmml.cli.run.md_system import build_parser

    psf = tmp_path / "model.psf"
    crd = tmp_path / "model.crd"
    ckpt = tmp_path / "ckpt.json"
    for path in (psf, crd, ckpt):
        path.write_text("x")
    args = build_parser().parse_args(
        [
            "--backend",
            "pycharmm",
            "--setup",
            "pbc_npt",
            "--from-psf",
            str(psf),
            "--from-crd",
            str(crd),
            "--skip-cluster-build",
            "--box-size",
            "34",
            "--composition",
            "ETOH:405",
            "--checkpoint",
            str(ckpt),
            "--mm-switch-width",
            "3.0",
            "--no-calculator-pre-minimize",
            "--no-charmm-pre-minimize",
            "--no-monomer-physnet-mini",
            "--no-mc-density-equalize",
            "--md-stages",
            "equi",
            "--output-dir",
            str(tmp_path / "out"),
            "--job-name",
            "cpt_ml_virial",
        ]
    )
    assert args.backend == "pycharmm"
    assert args.box_size == 34.0
    assert args.mm_switch_width == 3.0
    assert args.calculator_pre_minimize is False
    assert args.charmm_pre_minimize is False
    assert args.monomer_physnet_mini is False
    assert args.mc_density_equalize is False


@pytest.mark.parametrize(
    "prsi,atomic,strain,want",
    [
        (0.4, 800.0, 1.2, "H1"),
        (400.0, 410.0, 2.0, "H2"),
        (500.0, 510.0, 490.0, "H3"),
        (1.5, 2.0, 0.8, "H4"),
        (100.0, 400.0, 380.0, "ambiguous"),
    ],
)
def test_classify_decision_table(prsi, atomic, strain, want):
    got, _reason = classify_cpt_ml_virial_hypothesis(
        p_prsi_atm=prsi,
        p_atomic_atm=atomic,
        p_strain_atm=strain,
    )
    assert got == want
