"""Error metrics and loss-term weighting in the DCMNet loss module.

``dcmnet/loss.py`` defines what the model learns and sat at 8.6% coverage. That
is where the dipole unit bug lived, and silently zeroed loss terms are the same
class of problem: a change to the training objective that is invisible from the
outside because the function still returns three plausible numbers.

``pred_dipole`` itself is covered by ``test_dcmnet_dipole_units.py``; this file
covers the metrics around it and the loss weighting.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from mmml.models.dcmnet.dcmnet.loss import (  # noqa: E402
    esp_loss_eval,
    mean_absolute_error,
)


# --- mean_absolute_error ----------------------------------------------------
#
# The metric deliberately ignores entries whose *target* is zero: batches are
# padded to a fixed atom count and the padding is stored as 0. Averaging over
# the padding too would divide by a batch-shape-dependent denominator and make
# the reported MAE drift with padding rather than with model quality.


def test_mae_matches_a_hand_computed_value():
    pred = jnp.array([[1.0, 2.0, 3.0]])
    target = jnp.array([[1.5, 2.5, 3.5]])
    assert float(mean_absolute_error(pred, target, 1)) == pytest.approx(0.5, rel=1e-6)


def test_mae_is_zero_for_an_exact_prediction():
    x = jnp.array([[1.0, -2.0, 3.0]])
    assert float(mean_absolute_error(x, x, 1)) == pytest.approx(0.0, abs=1e-7)


def test_mae_ignores_padded_entries():
    """Padding (target 0) must not dilute the average."""
    pred = jnp.array([[1.0, 2.0, 99.0]])
    target = jnp.array([[1.5, 2.5, 0.0]])  # third entry is padding
    # Only the two real atoms contribute: mean(|0.5|, |0.5|) == 0.5
    assert float(mean_absolute_error(pred, target, 1)) == pytest.approx(0.5, rel=1e-6)


def test_mae_is_symmetric_in_its_arguments_magnitude():
    a = jnp.array([[1.0, 2.0]])
    b = jnp.array([[3.0, 5.0]])
    assert float(mean_absolute_error(a, b, 1)) == pytest.approx(
        float(mean_absolute_error(b, a, 1)), rel=1e-6
    )


def test_mae_scales_linearly_with_the_residual():
    target = jnp.array([[1.0, 2.0]])
    one = float(mean_absolute_error(target + 1.0, target, 1))
    two = float(mean_absolute_error(target + 2.0, target, 1))
    assert two == pytest.approx(2.0 * one, rel=1e-5)


# --- esp_loss_eval ----------------------------------------------------------
#
# Documented as an RMSE. optax.l2_loss is 0.5 * r^2, so the * 2 restores r^2
# before the mean and square root -- getting that factor wrong would understate
# the reported error by sqrt(2).


def test_esp_loss_eval_is_a_root_mean_square_error():
    target = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 4.0, 6.0])  # residuals 1, 2, 3

    got = esp_loss_eval(pred, target, None)

    assert got == pytest.approx(np.sqrt(np.mean([1.0, 4.0, 9.0])), rel=1e-9)


def test_esp_loss_eval_is_zero_for_an_exact_prediction():
    x = np.array([1.0, -2.0, 0.5])
    assert esp_loss_eval(x, x, None) == pytest.approx(0.0, abs=1e-12)


def test_esp_loss_eval_ignores_zero_targets():
    """Grid points masked out of the ESP target are stored as 0."""
    target = np.array([1.0, 0.0, 3.0])
    pred = np.array([2.0, 999.0, 5.0])  # the masked point is wildly wrong

    got = esp_loss_eval(pred, target, None)

    assert got == pytest.approx(np.sqrt(np.mean([1.0, 4.0])), rel=1e-9)


def test_esp_loss_eval_flattens_multidimensional_input():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = target + 1.0
    assert esp_loss_eval(pred.flatten(), target, None) == pytest.approx(1.0, rel=1e-9)


def test_esp_loss_eval_scales_linearly_with_the_residual():
    target = np.array([1.0, 2.0, 3.0])
    one = esp_loss_eval(target + 1.0, target, None)
    three = esp_loss_eval(target + 3.0, target, None)
    assert three == pytest.approx(3.0 * one, rel=1e-9)


# --- the zeroed loss terms --------------------------------------------------


@pytest.mark.parametrize("esp_w,chg_w", [(1.0, 1.0), (3.0, 5.0), (0.0, 0.0)])
def test_loss_terms_have_expected_values_and_independent_weights(esp_w, chg_w):
    """Evaluate the objective: source spelling cannot establish active loss terms."""
    from mmml.models.dcmnet.dcmnet.loss import dipo_esp_mono_loss

    # Zero charges give zero ESP and dipole. Nonzero targets make every loss
    # nonzero before weighting; the second grid point is deliberately masked.
    terms = dipo_esp_mono_loss(
        dipo_prediction=jnp.zeros((1, 2, 1, 3)),
        mono_prediction=jnp.zeros((1, 2, 1)),
        esp_target=jnp.array([[2.0, 999.0]]),
        vdw_surface=jnp.array([[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        mono=jnp.array([1.0, -1.0]),
        Dxyz=jnp.array([[1.0, 2.0, 3.0]]),
        com=jnp.zeros((1, 3)),
        espMask=jnp.array([[True, False]]),
        n_atoms=jnp.array([2]),
        batch_size=1,
        esp_w=esp_w,
        chg_w=chg_w,
        n_dcm=1,
    )
    np.testing.assert_allclose(np.asarray(terms), [2.0 * esp_w, 0.5, 7.0 * chg_w])
