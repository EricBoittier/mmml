"""run_chunked_model_apply: skip chunks that hold only padding past ``n_valid``."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot_gpu import run_chunked_model_apply

N_CHUNKS, CHUNK, MAX_ATOMS = 5, 4, 3
BATCH = 18  # last chunk is partly beyond the batch


def _inputs():
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.normal(size=(N_CHUNKS, CHUNK, MAX_ATOMS, 3)))
    z = jnp.ones((N_CHUNKS, CHUNK, MAX_ATOMS), dtype=jnp.int32)
    n = jnp.full((N_CHUNKS, CHUNK), MAX_ATOMS, dtype=jnp.int32)
    return r, z, n


def _apply_one(r, z, n):
    del z, n
    energy = jnp.sum(jnp.sin(r) ** 2, axis=(1, 2))
    forces = jnp.cos(r).reshape(-1, 3)
    charges = r[..., :1].reshape(-1, 1)
    return energy, forces, charges


def _run(n_valid, r, z, n):
    return run_chunked_model_apply(
        R_chunks=r,
        Z_chunks=z,
        N_chunks=n,
        n_chunks=N_CHUNKS,
        effective_batch_size=BATCH,
        chunk_size=CHUNK,
        max_atoms=MAX_ATOMS,
        n_gpus=1,
        apply_one_chunk=_apply_one,
        has_aux=True,
        n_valid=n_valid,
    )


@pytest.mark.parametrize("n_valid", [0, 1, 4, 5, 9, 17, 18])
def test_valid_slots_identical_and_padding_chunks_zero(n_valid):
    r, z, n = _inputs()
    e_ref, f_ref, q_ref = _run(None, r, z, n)
    e, f, q = _run(jnp.asarray(n_valid), r, z, n)
    assert e.shape == e_ref.shape and f.shape == f_ref.shape and q.shape == q_ref.shape
    # Every chunk that holds a valid slot is evaluated exactly as before.
    n_eval = -(-n_valid // CHUNK) * CHUNK
    np.testing.assert_array_equal(np.asarray(e[:n_eval]), np.asarray(e_ref[:n_eval]))
    np.testing.assert_array_equal(
        np.asarray(f[: n_eval * MAX_ATOMS]), np.asarray(f_ref[: n_eval * MAX_ATOMS])
    )
    np.testing.assert_array_equal(
        np.asarray(q[: n_eval * MAX_ATOMS]), np.asarray(q_ref[: n_eval * MAX_ATOMS])
    )
    # Chunks entirely past n_valid are skipped (zeros), not run on padding.
    assert not np.any(np.asarray(e[n_eval:]))
    assert not np.any(np.asarray(f[n_eval * MAX_ATOMS :]))


def test_skipped_chunks_do_not_run_the_model():
    r, z, n = _inputs()
    calls = []

    def counting_apply(rc, zc, nc):
        jax.debug.callback(lambda: calls.append(1))
        return _apply_one(rc, zc, nc)

    out = run_chunked_model_apply(
        R_chunks=r,
        Z_chunks=z,
        N_chunks=n,
        n_chunks=N_CHUNKS,
        effective_batch_size=BATCH,
        chunk_size=CHUNK,
        max_atoms=MAX_ATOMS,
        n_gpus=1,
        apply_one_chunk=counting_apply,
        has_aux=True,
        n_valid=jnp.asarray(6),
    )
    jax.block_until_ready(out)
    assert len(calls) == 2


def test_traced_n_valid_does_not_recompile():
    r, z, n = _inputs()
    traces = []

    @jax.jit
    def f(n_valid):
        traces.append(1)
        e, fo, _ = _run(n_valid, r, z, n)
        return jnp.sum(e) + jnp.sum(fo)

    vals = [float(f(jnp.asarray(k))) for k in (3, 11, 18, 7)]
    assert len(traces) == 1
    assert vals[2] == pytest.approx(
        float(jnp.sum(_run(None, r, z, n)[0][:BATCH]) + jnp.sum(_run(None, r, z, n)[1]))
    )


def test_gradients_flow_through_evaluated_chunks():
    r, z, n = _inputs()

    def loss(rr, n_valid):
        e, _, _ = _run(n_valid, rr, z, n)
        return jnp.sum(e[:n_valid_static])

    n_valid_static = 9
    g_skip = jax.grad(loss)(r, jnp.asarray(n_valid_static))
    g_full = jax.grad(loss)(r, None)
    np.testing.assert_allclose(np.asarray(g_skip), np.asarray(g_full), rtol=0, atol=0)


def _run_static(n_eval_chunks, r, z, n, apply_one=_apply_one):
    return run_chunked_model_apply(
        R_chunks=r,
        Z_chunks=z,
        N_chunks=n,
        n_chunks=N_CHUNKS,
        effective_batch_size=BATCH,
        chunk_size=CHUNK,
        max_atoms=MAX_ATOMS,
        n_gpus=1,
        apply_one_chunk=apply_one,
        has_aux=True,
        n_eval_chunks=n_eval_chunks,
    )


@pytest.mark.parametrize("n_eval", [1, 2, 3, 5, 9])
def test_static_chunk_budget_matches_cond_skip_bitwise(n_eval):
    """A static trip count gives the same outputs as the traced lax.cond skip."""
    r, z, n = _inputs()
    n_valid = min(n_eval, N_CHUNKS) * CHUNK
    ref = _run(jnp.asarray(n_valid), r, z, n)
    out = _run_static(n_eval, r, z, n)
    for a, b in zip(out, ref):
        assert a.shape == b.shape
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_static_chunk_budget_has_no_conditional_and_runs_only_budget_chunks():
    r, z, n = _inputs()
    calls = []

    def counting_apply(rc, zc, nc):
        jax.debug.callback(lambda: calls.append(1))
        return _apply_one(rc, zc, nc)

    jax.block_until_ready(_run_static(2, r, z, n, counting_apply))
    assert len(calls) == 2
    jaxpr = str(jax.make_jaxpr(lambda rr: _run_static(2, rr, z, n))(r))
    assert "cond[" not in jaxpr
    # the chunk loop is a scan with a compile-time length
    assert "length=2" in jaxpr


def test_static_chunk_budget_gradients_match_full():
    r, z, n = _inputs()

    def loss(rr, n_eval):
        e, _, _ = _run_static(n_eval, rr, z, n)
        return jnp.sum(e[:9])

    np.testing.assert_array_equal(
        np.asarray(jax.grad(loss)(r, 3)), np.asarray(jax.grad(loss)(r, N_CHUNKS))
    )
