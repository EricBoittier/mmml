"""Per-element reference subtraction: exactness, invariants, and refusal cases."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.cli.misc.subtract_element_refs import fit_element_refs, main


def _synthetic(n=200, seed=0):
    """E built as an exact per-element sum plus a small interaction term."""
    rng = np.random.default_rng(seed)
    Z = rng.choice([1, 6, 8], size=(n, 10))
    refs = {1: -0.5, 6: -37.8, 8: -75.1}
    base = np.array([sum(refs[int(z)] for z in row) for row in Z])
    inter = rng.normal(0, 0.01, size=n)  # the real signal
    return Z, base + inter, refs, inter


def test_fit_recovers_known_reference_energies():
    Z, E, refs, _ = _synthetic()
    zs, coef, _ = fit_element_refs(E, Z)
    got = dict(zip(zs.tolist(), coef.tolist()))
    for z, want in refs.items():
        assert got[z] == pytest.approx(want, abs=1e-3), f"element {z}"


def test_residual_is_the_interaction_term():
    """The residual must *be* the interaction signal, not a rescaled shadow.

    Not an exact match: with a finite sample the least-squares fit absorbs a
    little of the interaction term into the coefficients, so the test asserts
    near-perfect correlation and a preserved scale rather than equality.
    """
    Z, E, _, inter = _synthetic(n=2000)
    zs, coef, counts = fit_element_refs(E, Z)
    residual = E - counts @ coef

    r = np.corrcoef(residual, inter)[0, 1]
    assert r > 0.99, f"residual does not track the true interaction (r={r:.4f})"
    assert residual.std() == pytest.approx(inter.std(), rel=0.05)


def test_residual_variance_is_much_smaller_than_raw():
    Z, E, _, _ = _synthetic()
    zs, coef, counts = fit_element_refs(E, Z)
    residual = E - counts @ coef
    assert residual.var() < 1e-3 * E.var()


def test_padding_atoms_excluded_from_the_fit():
    """Z=0 padding must not become a fitted 'element'."""
    Z, E, _, _ = _synthetic()
    Z = np.concatenate([Z, np.zeros((len(Z), 5), dtype=int)], axis=1)
    zs, _, _ = fit_element_refs(E, Z)
    assert 0 not in zs.tolist()


def _write(tmp_path, Z, E, name="in.npz"):
    n = len(E)
    p = tmp_path / name
    np.savez(p, E=E.reshape(n, 1), Z=Z, R=np.zeros((n, Z.shape[1], 3)),
             F=np.zeros((n, Z.shape[1], 3)), N=np.full((n, 1), Z.shape[1]),
             cgenff_master_sigmas=np.arange(185, dtype=float))
    return p


def test_writes_shifted_energies_and_keeps_the_fit(tmp_path):
    Z, E, _, _ = _synthetic()
    inp = _write(tmp_path, Z, E)
    out = tmp_path / "out.npz"
    assert main(["--in", str(inp), "--out", str(out)]) == 0

    d = np.load(out)
    assert d["E"].std() < 0.1 * E.std(), "energies should be much smaller now"
    assert set(d["element_ref_Z"].tolist()) == {1, 6, 8}
    assert len(d["element_ref_E_eV"]) == 3


def test_shift_is_exactly_recoverable(tmp_path):
    """The stored coefficients must reconstruct the original energies."""
    Z, E, _, _ = _synthetic()
    inp = _write(tmp_path, Z, E)
    out = tmp_path / "out.npz"
    main(["--in", str(inp), "--out", str(out)])

    d = np.load(out)
    zs, coef = d["element_ref_Z"], d["element_ref_E_eV"]
    counts = np.stack([(d["Z"] == z).sum(axis=1) for z in zs], axis=1).astype(float)
    restored = d["E"].ravel() + counts @ coef
    np.testing.assert_allclose(restored, E, atol=1e-8)


def test_forces_are_untouched(tmp_path):
    """A per-composition constant cannot change forces."""
    Z, E, _, _ = _synthetic()
    inp = _write(tmp_path, Z, E)
    out = tmp_path / "out.npz"
    main(["--in", str(inp), "--out", str(out)])
    np.testing.assert_array_equal(np.load(out)["F"], np.load(inp)["F"])


def test_refuses_when_composition_explains_little(tmp_path):
    """If composition is not the dominant term, this is the wrong transform."""
    rng = np.random.default_rng(1)
    Z = np.full((200, 4), 6)          # identical composition everywhere
    E = rng.normal(0, 1.0, size=200)  # variance is pure signal
    inp = _write(tmp_path, Z, E)
    out = tmp_path / "out.npz"
    with pytest.raises(SystemExit, match="explains only"):
        main(["--in", str(inp), "--out", str(out)])


def test_refuses_to_overwrite_input(tmp_path):
    Z, E, _, _ = _synthetic()
    inp = _write(tmp_path, Z, E)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--in", str(inp), "--out", str(inp)])


def test_dry_run_writes_nothing(tmp_path):
    Z, E, _, _ = _synthetic()
    inp = _write(tmp_path, Z, E)
    out = tmp_path / "out.npz"
    assert main(["--in", str(inp), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
