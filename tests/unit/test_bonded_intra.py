"""Unit tests for --ml-potential-mode bonded_intra.

See docs/hybrid-bonded-intra.md. The mode hands the internal monomer energy to
CGenFF bonded while the ML model keeps the dimer interaction.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_build_bonded_intra_evaluator_rejects_heterogeneous_monomers() -> None:
    """The batched path slices one static width, so mixed sizes must not proceed."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        build_bonded_intra_evaluator,
    )

    with pytest.raises(NotImplementedError, match="homogeneous monomers"):
        build_bonded_intra_evaluator(
            atoms_per_monomer_list=[3, 3, 5],
            monomer_psf="/nonexistent.psf",
        )


def test_build_bonded_intra_evaluator_requires_a_psf() -> None:
    """Without a PSF the evaluator silently becomes a minimal harmonic chain.

    That is not a water potential, and the failure would be invisible in the
    energies -- so it must raise rather than degrade.
    """
    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        build_bonded_intra_evaluator,
    )

    with pytest.raises(ValueError, match="needs a PSF"):
        build_bonded_intra_evaluator(
            atoms_per_monomer_list=[3, 3],
            monomer_psf=None,
        )


def test_bonded_intra_contribution_sums_and_scatters() -> None:
    """Energies sum over monomers; forces land on the right global atoms.

    Padding columns in ``monomer_idx`` must never reach the bonded terms, so the
    index array here is deliberately wider than the monomer.
    """
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        bonded_intra_contribution,
    )

    n_monomers, atoms_per, total_atoms = 2, 3, 6
    positions = jnp.asarray(np.arange(total_atoms * 3, dtype=np.float64).reshape(-1, 3))
    # Column 3 is padding pointing at a sentinel row that must stay untouched.
    monomer_idx = jnp.asarray([[0, 1, 2, 5], [3, 4, 5, 5]], dtype=jnp.int32)

    def fake_eval(pos):
        # Energy distinguishes the monomers; forces are unique per atom so a
        # mis-scatter cannot pass by coincidence.
        return jnp.sum(pos), pos * 2.0

    energy, forces = bonded_intra_contribution(
        fake_eval, positions, monomer_idx, atoms_per, total_atoms
    )

    expected_energy = float(jnp.sum(positions[:6]))
    assert float(energy) == pytest.approx(expected_energy)

    # Every real atom appears exactly once, so forces are just 2*position.
    np.testing.assert_allclose(np.asarray(forces), np.asarray(positions) * 2.0)
    assert np.asarray(forces).shape == (total_atoms, 3)


def test_bonded_intra_damping_brackets_the_measured_regimes() -> None:
    """Full strength for thermal distortion, zero where the noise well sits.

    Anchored on measured bonded energies (docs/hybrid-bonded-intra.md): ~0.3
    kcal/mol for ordinary thermal O-H fluctuation, +15.3 kcal/mol at O-H =
    0.771 A where the interaction term reads -73.5 kcal/mol of pure noise.
    """
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import bonded_intra_damping

    kcal = 1.0 / 23.060548
    e = jnp.asarray([0.0, 0.3, 5.0, 10.0, 15.3, 40.0]) * kcal
    s, _ = bonded_intra_damping(e, onset_kcal=5.0, cutoff_kcal=15.0)
    s = np.asarray(s)

    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(1.0), "thermal distortion must not be damped"
    assert s[2] == pytest.approx(1.0), "onset is inclusive"
    assert 0.0 < s[3] < 1.0, "mid-taper"
    assert s[4] == pytest.approx(0.0), "the noise well must be fully damped"
    assert s[5] == pytest.approx(0.0)
    assert np.all(np.diff(s) <= 1e-12), "damping must be monotone in distortion"


def test_bonded_intra_damping_derivative_matches_finite_differences() -> None:
    """ds/dE must be exact: forces here are assembled by hand, not autodiffed.

    A wrong derivative would not change any energy -- it would silently make the
    forces non-conservative, which is the failure this whole guard exists to
    prevent. Checked against central differences rather than against the closed
    form it was derived from.
    """
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import bonded_intra_damping

    kcal = 1.0 / 23.060548
    onset, cutoff = 5.0, 15.0
    h = 1e-6 * kcal
    for e_kcal in (6.0, 8.0, 10.0, 12.0, 14.0):
        e = jnp.asarray([e_kcal * kcal])
        _, dsde = bonded_intra_damping(e, onset_kcal=onset, cutoff_kcal=cutoff)
        s_plus, _ = bonded_intra_damping(e + h, onset_kcal=onset, cutoff_kcal=cutoff)
        s_minus, _ = bonded_intra_damping(e - h, onset_kcal=onset, cutoff_kcal=cutoff)
        fd = float((s_plus[0] - s_minus[0]) / (2 * h))
        assert float(dsde[0]) == pytest.approx(fd, rel=1e-4), f"at {e_kcal} kcal/mol"

    # Outside the taper the derivative must be exactly zero, or the clip's kink
    # leaks a spurious force.
    for e_kcal in (0.0, 1.0, 20.0, 100.0):
        _, dsde = bonded_intra_damping(
            jnp.asarray([e_kcal * kcal]), onset_kcal=onset, cutoff_kcal=cutoff
        )
        assert float(dsde[0]) == 0.0


def test_md_system_parser_accepts_bonded_intra() -> None:
    from mmml.cli.run.md_system import build_parser

    args = build_parser().parse_args(
        ["--ml-potential-mode", "bonded_intra", "--jax-mm-spoof-psf", "/tmp/x.psf"]
    )
    assert args.ml_potential_mode == "bonded_intra"
    assert str(args.jax_mm_spoof_psf) == "/tmp/x.psf"

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ml-potential-mode", "not-a-mode"])


def test_factory_mmml_forwards_bonded_intra_configuration() -> None:
    """The mode is useless if the factory drops it before setup_calculator."""
    import inspect

    from mmml.cli.run.md_pbc_suite.ase import _factory_mmml

    params = inspect.signature(_factory_mmml).parameters
    assert "ml_potential_mode" in params
    assert "jax_mm_spoof_psf" in params
