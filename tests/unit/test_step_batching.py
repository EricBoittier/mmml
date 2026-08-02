"""Block stepping must be numerically identical to the Python step loop.

The point of :func:`make_block_stepper` is to trade N dispatches for one without
changing trajectories, and to make dynamic kwargs (neighbor lists) *arguments*
so a refresh between blocks actually reaches the integrator.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mmml.md.step_batching import make_block_stepper


def _damped_step(state, *, force_scale=1.0):
    x, v = state
    a = -x * force_scale
    v = v + 0.01 * a
    return (x + 0.01 * v, v)


def test_block_matches_python_loop_exactly():
    state = (jnp.array([1.0, -2.0]), jnp.array([0.0, 0.5]))

    ref = state
    for _ in range(16):
        ref = _damped_step(ref)

    got = make_block_stepper(_damped_step, 16)(state)

    assert jnp.allclose(got[0], ref[0], atol=0, rtol=0)
    assert jnp.allclose(got[1], ref[1], atol=0, rtol=0)


def test_dynamic_kwargs_are_arguments_not_baked_constants():
    """A changed kwarg must change the result without a fresh stepper.

    This is the failure mode that made λ-dynamics' neighbor refresh a no-op:
    reading mutable state inside a jitted body freezes it at trace time.
    """
    block = make_block_stepper(_damped_step, 4)
    state = (jnp.array([1.0]), jnp.array([0.0]))

    weak = block(state, force_scale=1.0)
    strong = block(state, force_scale=50.0)

    assert not jnp.allclose(weak[1], strong[1]), (
        "force_scale was baked in; dynamic kwargs must be traced as arguments"
    )


def test_normalize_is_traced_into_the_carry_and_the_body():
    counter = {"n": 0}

    def counting_normalize(s):
        counter["n"] += 1
        return s

    make_block_stepper(_damped_step, 5, normalize=counting_normalize)(
        (jnp.array([1.0]), jnp.array([0.0]))
    )
    # fori_loop traces its body once regardless of trip count, so normalize is
    # *traced* twice (initial carry + body) while *running* on every iteration.
    assert counter["n"] == 2


def test_normalize_holds_the_carry_dtype_across_the_loop():
    """fori_loop rejects a carry whose dtype changes; normalize is the fix."""

    def upcasting_step(state):
        # float32 in, float64 out under x64 -- an invalid fori_loop carry.
        return jnp.asarray(state, dtype=jnp.float64) + 1.0

    start = jnp.asarray([1.0], dtype=jnp.float32)

    out = make_block_stepper(
        upcasting_step, 3, normalize=lambda s: jnp.asarray(s, dtype=jnp.float32)
    )(start)
    assert out.dtype == jnp.float32
    assert jnp.allclose(out, jnp.asarray([4.0], dtype=jnp.float32))


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonpositive_block_size(bad):
    with pytest.raises(ValueError, match="block_steps must be >= 1"):
        make_block_stepper(_damped_step, bad)


def test_single_step_block_is_the_plain_step():
    state = (jnp.array([0.7]), jnp.array([-0.3]))
    got = make_block_stepper(_damped_step, 1)(state)
    ref = _damped_step(state)
    assert jnp.allclose(got[0], ref[0])
    assert jnp.allclose(got[1], ref[1])
