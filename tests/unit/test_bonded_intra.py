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


def test_resolve_bonded_intra_damping_defaults_to_off() -> None:
    """Silence here means existing bonded_intra runs keep their exact behaviour."""
    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        resolve_bonded_intra_damping,
    )

    onset, cutoff, label = resolve_bonded_intra_damping(None, 15.0)
    assert onset is None
    assert cutoff == 15.0
    assert "off" in label

    onset, cutoff, label = resolve_bonded_intra_damping(5.0, 15.0)
    assert (onset, cutoff) == (5.0, 15.0)
    assert "5->15 kcal/mol" in label

    # An inverted window would make the taper run backwards -- amplifying the
    # interaction exactly where it is noise -- so it must not be accepted.
    with pytest.raises(ValueError, match="must exceed"):
        resolve_bonded_intra_damping(15.0, 5.0)
    with pytest.raises(ValueError, match="must exceed"):
        resolve_bonded_intra_damping(5.0, 5.0)


def test_bonded_intra_bundle_matches_the_unbundled_pieces() -> None:
    """The bundle is glue; it must not change the energy or force it glues."""
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        bonded_intra_bundle,
        bonded_intra_contribution,
        bonded_intra_damping,
        bonded_intra_per_monomer,
    )

    atoms_per, total_atoms = 3, 6
    positions = jnp.asarray(
        np.linspace(0.0, 2.0, total_atoms * 3, dtype=np.float64).reshape(-1, 3)
    )
    monomer_idx = jnp.asarray([[0, 1, 2, 5], [3, 4, 5, 5]], dtype=jnp.int32)

    def fake_eval(pos):
        return jnp.sum(pos**2), pos * 3.0

    ref_e, ref_f = bonded_intra_contribution(
        fake_eval, positions, monomer_idx, atoms_per, total_atoms
    )

    e, f, damping = bonded_intra_bundle(
        fake_eval, positions, monomer_idx, atoms_per, total_atoms
    )
    assert damping is None, "damping must stay off unless an onset is given"
    assert float(e) == pytest.approx(float(ref_e))
    np.testing.assert_allclose(np.asarray(f), np.asarray(ref_f))

    e, f, damping = bonded_intra_bundle(
        fake_eval,
        positions,
        monomer_idx,
        atoms_per,
        total_atoms,
        damp_onset_kcal=5.0,
        damp_cutoff_kcal=15.0,
    )
    assert float(e) == pytest.approx(float(ref_e)), "damping must not touch the total"
    np.testing.assert_allclose(np.asarray(f), np.asarray(ref_f))

    e_mono, f_mono = bonded_intra_per_monomer(
        fake_eval, positions, monomer_idx, atoms_per
    )
    exp_s, exp_dsde = bonded_intra_damping(e_mono, onset_kcal=5.0, cutoff_kcal=15.0)
    s, dsde, f_bonded = damping
    np.testing.assert_allclose(np.asarray(s), np.asarray(exp_s))
    np.testing.assert_allclose(np.asarray(dsde), np.asarray(exp_dsde))
    np.testing.assert_allclose(np.asarray(f_bonded), np.asarray(f_mono))


@pytest.fixture
def jax_x64():
    """float64 for the duration of one test, restored afterwards.

    Central differences on a smoothstep are not resolvable in float32, and the
    flag is process-wide in JAX, so it has to be put back.
    """
    import jax

    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _toy_damped_system():
    """A 4-monomer toy spanning every damping regime, with analytic energies.

    Bond lengths are picked so the monomers land below onset, twice inside the
    taper, and past cutoff -- dimer (1, 2) has *both* monomers mid-taper, which
    is the only configuration where both product-rule terms are live at once.
    """
    import jax.numpy as jnp

    apm, n_mono = 3, 4
    onset_kcal, cutoff_kcal = 5.0, 15.0
    bond_lengths = [1.00, 1.13, 1.17, 1.30]

    mono_idx = jnp.arange(n_mono * apm, dtype=jnp.int32).reshape(n_mono, apm)
    pairs = jnp.asarray(
        [[a, b] for a in range(n_mono) for b in range(a + 1, n_mono)], dtype=jnp.int32
    )
    # Dimer layout is monomer A's atoms then monomer B's, per indices_of_pairs.
    dimer_idx = jnp.concatenate([mono_idx[pairs[:, 0]], mono_idx[pairs[:, 1]]], axis=1)

    rows = []
    for m, d in enumerate(bond_lengths):
        base = np.array([4.0 * m, 0.0, 0.0])
        rows += [base, base + [d, 0.0, 0.0], base + [0.0, d, 0.0]]
    positions = jnp.asarray(np.asarray(rows, dtype=np.float64))

    def e_bonded(x):
        """Two harmonic bonds off atom 0; ~a water skeleton, in eV."""
        d01 = jnp.linalg.norm(x[1] - x[0])
        d02 = jnp.linalg.norm(x[2] - x[0])
        return 10.0 * ((d01 - 1.0) ** 2 + (d02 - 1.0) ** 2)

    def e_int(x):
        """Coulomb-like A-B interaction over the dimer's two halves."""
        a, b = x[:apm], x[apm:]
        r = jnp.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return 0.5 * jnp.sum(1.0 / r)

    return dict(
        apm=apm,
        n_mono=n_mono,
        total_atoms=n_mono * apm,
        mono_idx=mono_idx,
        pairs=pairs,
        dimer_idx=dimer_idx,
        positions=positions,
        e_bonded=e_bonded,
        e_int=e_int,
        onset_kcal=onset_kcal,
        cutoff_kcal=cutoff_kcal,
    )


def _assemble(sys_, positions):
    """Damped dimer energy + global forces, built the way the calculator builds them."""
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import (
        apply_bonded_intra_damping,
        bonded_intra_damping,
    )

    e_bonded_mono, grad_bonded = jax.vmap(jax.value_and_grad(sys_["e_bonded"]))(
        positions[sys_["mono_idx"]]
    )
    s_mono, dsde_mono = bonded_intra_damping(
        e_bonded_mono, onset_kcal=sys_["onset_kcal"], cutoff_kcal=sys_["cutoff_kcal"]
    )
    e_int, grad_int = jax.vmap(jax.value_and_grad(sys_["e_int"]))(
        positions[sys_["dimer_idx"]]
    )
    mask = jnp.ones(sys_["dimer_idx"].shape, dtype=bool)

    e_damped, f_damped = apply_bonded_intra_damping(
        e_int,
        -grad_int,
        sys_["pairs"],
        s_mono,
        dsde_mono,
        -grad_bonded,
        mask,
    )
    forces = jax.ops.segment_sum(
        f_damped.reshape(-1, 3),
        sys_["dimer_idx"].reshape(-1),
        num_segments=sys_["total_atoms"],
    )
    return e_damped, forces, s_mono, dsde_mono


def _total_damped_energy(sys_, positions):
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mmml_calculator import bonded_intra_damping

    e_bonded_mono = jax.vmap(sys_["e_bonded"])(positions[sys_["mono_idx"]])
    s, _ = bonded_intra_damping(
        e_bonded_mono, onset_kcal=sys_["onset_kcal"], cutoff_kcal=sys_["cutoff_kcal"]
    )
    e_int = jax.vmap(sys_["e_int"])(positions[sys_["dimer_idx"]])
    pairs = sys_["pairs"]
    return jnp.sum(s[pairs[:, 0]] * s[pairs[:, 1]] * e_int)


def test_toy_system_spans_the_damping_regimes(jax_x64) -> None:
    """Guards the fixture: if every monomer sat outside the taper the force test
    below would pass with the product-rule terms deleted."""
    sys_ = _toy_damped_system()
    _, _, s_mono, dsde_mono = _assemble(sys_, sys_["positions"])
    s = np.asarray(s_mono)
    dsde = np.asarray(dsde_mono)

    assert s[0] == pytest.approx(1.0), "monomer 0 must sit below onset"
    assert 0.0 < s[1] < 1.0 and dsde[1] != 0.0, "monomer 1 must be mid-taper"
    assert 0.0 < s[2] < 1.0 and dsde[2] != 0.0, "monomer 2 must be mid-taper"
    assert s[3] == pytest.approx(0.0), "monomer 3 must be past cutoff"


def test_assembled_damped_forces_match_autodiff(jax_x64) -> None:
    """The hand-assembled damped force must equal -grad of the damped energy.

    This is the gate the primitive's own derivative test cannot provide: it
    covers the product rule, the sign of ds/dR = -(ds/dE) F_bonded, and the
    mapping of each monomer's bonded force onto its half of the dimer layout.
    Nothing in the assembled path uses autodiff, so jax.grad is an independent
    reference here.
    """
    import jax

    sys_ = _toy_damped_system()
    pos = sys_["positions"]

    e_damped, forces, _, _ = _assemble(sys_, pos)
    reference = -jax.grad(lambda p: _total_damped_energy(sys_, p))(pos)

    assert float(e_damped.sum()) == pytest.approx(
        float(_total_damped_energy(sys_, pos)), rel=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(forces), np.asarray(reference), rtol=1e-9, atol=1e-11
    )


def test_damped_total_energy_matches_central_differences(jax_x64) -> None:
    """Anchors the autodiff reference itself against finite differences."""
    import jax

    sys_ = _toy_damped_system()
    pos = sys_["positions"]
    _, forces, _, _ = _assemble(sys_, pos)

    h = 1e-6
    # Atoms of the two mid-taper monomers, where both product-rule terms are live.
    for atom in (3, 4, 5, 6, 7, 8):
        for axis in range(3):
            plus = pos.at[atom, axis].add(h)
            minus = pos.at[atom, axis].add(-h)
            fd = -float(
                (_total_damped_energy(sys_, plus) - _total_damped_energy(sys_, minus))
                / (2 * h)
            )
            assert float(forces[atom, axis]) == pytest.approx(
                fd, rel=1e-5, abs=1e-7
            ), f"atom {atom}, axis {axis}"


def test_dropping_the_product_rule_term_would_be_caught(jax_x64) -> None:
    """The ds/dR term must dominate here, or the force test above is vacuous."""
    import jax

    sys_ = _toy_damped_system()
    pos = sys_["positions"]
    _, forces, s_mono, _ = _assemble(sys_, pos)

    e_int = jax.vmap(sys_["e_int"])(pos[sys_["dimer_idx"]])
    grad_int = jax.vmap(jax.grad(sys_["e_int"]))(pos[sys_["dimer_idx"]])
    pairs = sys_["pairs"]
    scale = s_mono[pairs[:, 0]] * s_mono[pairs[:, 1]]
    naive = jax.ops.segment_sum(
        (-grad_int * scale[:, None, None]).reshape(-1, 3),
        sys_["dimer_idx"].reshape(-1),
        num_segments=sys_["total_atoms"],
    )
    del e_int

    correct_norm = float(np.linalg.norm(np.asarray(forces)))
    naive_norm = float(np.linalg.norm(np.asarray(forces) - np.asarray(naive)))
    assert naive_norm > 0.1 * correct_norm, (
        "the ds/dR contribution is too small in this fixture for the "
        "force test to be meaningful"
    )


def test_full_damping_zeroes_a_dimer(jax_x64) -> None:
    """A monomer past cutoff must remove its dimers' energy and force entirely."""
    sys_ = _toy_damped_system()
    e_damped, _, _, _ = _assemble(sys_, sys_["positions"])

    pairs = np.asarray(sys_["pairs"])
    for d, (a, b) in enumerate(pairs):
        if 3 in (a, b):  # monomer 3 sits past cutoff
            assert float(e_damped[d]) == pytest.approx(0.0, abs=1e-14)


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
