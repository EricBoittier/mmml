"""``ReactionChannelRestraint``: a flat bottom that follows the reference path.

The property that matters for the analysis is not the functional form but that
the restraint is **one fixed function of the coordinates** — it interpolates at
the configuration's *own* xi, never at a window's target — so it is identical in
every window and cancels in the MBAR reduced-potential differences exactly as
the other walls do. Aimed at each window's xi0 instead it would not cancel and
would force a two-dimensional MBAR.
``test_costs_nothing_anywhere_along_the_reference_path`` and
``test_is_a_function_of_configuration_only`` are that property.

Geometry throughout: three collinear atoms, C at the origin with X at +a and N
at -b, so ``r(C,X) = a``, ``r(C,N) = b``, ``xi = a - b`` and ``sum = a + b`` are
set exactly rather than fitted.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from mmml.md.restraints import LinearDistanceCV, ReactionChannelRestraint  # noqa: E402

# A V-shaped reference path: the sum contracts toward the transition state.
XI_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)
SUM_GRID = (4.4, 4.0, 3.6, 4.0, 4.4)
K = 50.0
TOL = 0.3

CV_XI = LinearDistanceCV(pairs=((0, 1), (0, 2)), coefficients=(1.0, -1.0))
CV_SUM = LinearDistanceCV(pairs=((0, 1), (0, 2)), coefficients=(1.0, 1.0))


def _channel(**kw) -> ReactionChannelRestraint:
    return ReactionChannelRestraint(
        cv_xi=CV_XI, cv_sum=CV_SUM, xi_grid=XI_GRID, sum_grid=SUM_GRID,
        k=kw.pop("k", K), tol=kw.pop("tol", TOL), **kw
    )


def _geometry(xi: float, total: float) -> np.ndarray:
    """Collinear C/X/N with the requested ``xi`` and ``sum``."""
    a = (total + xi) / 2.0
    b = (total - xi) / 2.0
    assert a > 0 and b > 0, "unphysical request"
    return np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0], [-b, 0.0, 0.0]])


def _ref_sum(xi: float) -> float:
    return float(np.interp(xi, XI_GRID, SUM_GRID))


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kw, message",
    [
        (dict(xi_grid=(0.0, 1.0), sum_grid=(3.0,)), "same length"),
        (dict(xi_grid=(0.0,), sum_grid=(3.0,)), "at least two points"),
        (dict(xi_grid=(1.0, 0.0), sum_grid=(3.0, 3.0)), "increasing"),
        (dict(k=-1.0), "non-negative"),
        (dict(tol=-0.1), "non-negative"),
    ],
)
def test_rejects_a_channel_it_cannot_interpolate(kw, message):
    base = dict(cv_xi=CV_XI, cv_sum=CV_SUM, xi_grid=XI_GRID, sum_grid=SUM_GRID,
                k=K, tol=TOL)
    base.update(kw)
    with pytest.raises(ValueError, match=message):
        ReactionChannelRestraint(**base)


# ------------------------------------------------------------- the flat bottom


@pytest.mark.parametrize("xi", [-2.0, -1.5, -1.0, 0.0, 0.7, 1.0, 2.0])
def test_costs_nothing_anywhere_along_the_reference_path(xi):
    """Zero on the path itself, at every xi -- including between grid points."""
    ch = _channel()
    e = float(ch.energy(_geometry(xi, _ref_sum(xi))))
    assert e == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("offset", [-0.29, -0.1, 0.0, 0.1, 0.29])
def test_costs_nothing_inside_the_tolerance(offset):
    ch = _channel()
    xi = 0.5
    e = float(ch.energy(_geometry(xi, _ref_sum(xi) + offset)))
    assert e == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("excess", [0.1, 0.4, 1.0])
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_is_harmonic_in_the_excess_beyond_the_tolerance(excess, sign):
    ch = _channel()
    xi = 0.5
    deviation = sign * (TOL + excess)
    e = float(ch.energy(_geometry(xi, _ref_sum(xi) + deviation)))
    assert e == pytest.approx(0.5 * K * excess**2, rel=1e-9)


def test_penalty_is_symmetric_in_the_sign_of_the_deviation():
    ch = _channel()
    xi = -0.4
    hi = float(ch.energy(_geometry(xi, _ref_sum(xi) + TOL + 0.5)))
    lo = float(ch.energy(_geometry(xi, _ref_sum(xi) - TOL - 0.5)))
    assert hi == pytest.approx(lo, rel=1e-9)


# ------------------------------------------------- the MBAR-cancellation property


def test_is_a_function_of_configuration_only():
    """No window target enters anywhere, which is why it cancels in MBAR.

    Two restraints built independently, and a configuration evaluated by each,
    must agree: there is no per-window state for them to disagree about.
    """
    a, b = _channel(), _channel()
    for xi, total in [(-1.3, 4.9), (0.0, 3.6), (0.8, 4.6), (1.9, 3.2)]:
        R = _geometry(xi, total)
        assert float(a.energy(R)) == pytest.approx(float(b.energy(R)), rel=1e-12)


def test_moving_along_the_path_is_free_while_leaving_it_is_not():
    """The restraint bounds the sum, never the reaction coordinate itself."""
    ch = _channel()
    on_path = [float(ch.energy(_geometry(xi, _ref_sum(xi)))) for xi in XI_GRID]
    assert all(e == pytest.approx(0.0, abs=1e-12) for e in on_path)

    off_path = float(ch.energy(_geometry(0.0, _ref_sum(0.0) + 1.5)))
    assert off_path > 1.0


def test_value_numpy_reports_the_signed_deviation_from_the_channel_centre():
    ch = _channel()
    xi = 0.25
    for delta in (-0.8, 0.0, 0.6):
        got = ch.value_numpy(_geometry(xi, _ref_sum(xi) + delta))
        assert got == pytest.approx(delta, abs=1e-9)


# ------------------------------------------------------------- specs and wiring


def test_spec_round_trip_preserves_the_channel():
    ch = _channel()
    back = ReactionChannelRestraint.from_spec(ch.to_spec())
    assert back.xi_grid == ch.xi_grid
    assert back.sum_grid == ch.sum_grid
    assert back.k == ch.k and back.tol == ch.tol
    assert back.cv_xi.pairs == ch.cv_xi.pairs
    assert back.cv_sum.coefficients == ch.cv_sum.coefficients

    R = _geometry(0.3, 4.9)
    assert float(back.energy(R)) == pytest.approx(float(ch.energy(R)), rel=1e-12)


def test_from_spec_is_idempotent_on_an_instance():
    ch = _channel()
    assert ReactionChannelRestraint.from_spec(ch) is ch


def test_validate_against_checks_both_cvs():
    ch = _channel()
    ch.validate_against(3)  # all indices in range
    with pytest.raises((ValueError, IndexError)):
        ch.validate_against(2)  # atom index 2 is now out of range


@pytest.mark.parametrize(
    "module", ["mmml.umbrella.config", "mmml.umbrella.energy"]
)
def test_both_copies_of_resolve_wall_know_the_channel(module):
    """``umbrella.energy._resolve_wall`` duplicates ``umbrella.config``'s.

    Its own docstring warns that teaching only the config copy gets a spec past
    argument parsing and then fails inside the sampler, so both are asserted.
    """
    import importlib

    resolve = importlib.import_module(module)._resolve_wall
    spec = _channel().to_spec()
    assert isinstance(resolve(spec), ReactionChannelRestraint)
    # An already-built instance passes through untouched.
    ch = _channel()
    assert resolve(ch) is ch


def test_label_names_the_restrained_cv_and_the_channel_extent():
    label = _channel().label()
    assert "-2" in label and "+2" in label
    assert "50" in label


# --------------------------------------------------------------- batched paths


def test_forces_batched_is_minus_the_gradient_of_energy_batched():
    import jax.numpy as jnp

    ch = _channel()
    n_windows, n_atoms = 3, 3
    flat = jnp.asarray(np.concatenate([
        _geometry(-0.5, _ref_sum(-0.5) + 0.9),
        _geometry(0.0, _ref_sum(0.0)),
        _geometry(0.9, _ref_sum(0.9) - 1.1),
    ]))

    forces = np.asarray(ch.forces_batched(flat, n_atoms, n_windows))

    import jax

    def total(x):
        return ch.energy_batched(x, n_atoms, n_windows).sum()

    expected = -np.asarray(jax.grad(total)(flat)).reshape(n_windows * n_atoms, 3)
    np.testing.assert_allclose(forces, expected, atol=1e-9)


def test_energy_batched_matches_the_single_configuration_energy():
    import jax.numpy as jnp

    ch = _channel()
    configs = [
        _geometry(-0.5, _ref_sum(-0.5) + 0.9),
        _geometry(0.0, _ref_sum(0.0)),
        _geometry(0.9, _ref_sum(0.9) - 1.1),
    ]
    batched = np.asarray(ch.energy_batched(jnp.asarray(np.concatenate(configs)), 3, 3))
    singles = np.array([float(ch.energy(c)) for c in configs])
    np.testing.assert_allclose(batched.reshape(-1), singles, atol=1e-9)


def test_a_window_sitting_in_the_channel_contributes_no_force():
    """Flat bottom means the restraint is inert where sampling should be free."""
    import jax.numpy as jnp

    ch = _channel()
    on_path = jnp.asarray(_geometry(0.4, _ref_sum(0.4)))
    forces = np.asarray(ch.forces_batched(on_path, 3, 1))
    np.testing.assert_allclose(forces, 0.0, atol=1e-10)
