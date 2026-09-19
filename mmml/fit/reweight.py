"""Boltzmann reweighting of stored frames to new parameters (differentiable).

Frames x_i ~ exp(-beta U_0) (NVT, or NPT with the box stored per frame so the
PV term is theta-independent and cancels). For parameters theta:

    w_i(theta) = softmax(-beta [U_theta(x_i) - U_0(x_i)])
    <O>_theta  = sum_i w_i O_theta(x_i)

``jax.grad`` of a loss built on these averages gives the DiffTRe gradient
-beta Cov(O, dU/dtheta) + <dO/dtheta> without unrolling dynamics. Energies in
kcal/mol.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

KB_KCAL_MOL_K = 0.0019872041


def beta_kcal(temperature_K: float) -> float:
    return 1.0 / (KB_KCAL_MOL_K * float(temperature_K))


def reweight_weights(
    u_theta: jnp.ndarray, u_ref: jnp.ndarray, temperature_K: float
) -> jnp.ndarray:
    """Normalised weights of reference frames under ``u_theta`` (both (N,))."""
    return jax.nn.softmax(-beta_kcal(temperature_K) * (u_theta - u_ref))


def effective_sample_fraction(weights: jnp.ndarray) -> jnp.ndarray:
    """Kish effective sample size divided by N (1 = no reweighting)."""
    return 1.0 / (jnp.sum(weights**2) * weights.shape[0])


def reweighted_mean(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """``sum_i w_i O_i`` over the leading (frame) axis."""
    return jnp.tensordot(weights, values, axes=(0, 0))
