"""Host-side static chunk budget for the sparse-dimer PhysNet forward."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator
from mmml.interfaces.pycharmmInterface.mlpot.ml_chunk_budget import MlChunkBudget, MlChunkLayout

# ETOH:181 in a 26 A box: 181 monomers + 4005 dimer slots in 17 x 256 chunks.
LAYOUT = MlChunkLayout(n_monomers=181, max_active_dimers=4005, chunk_size=256, n_chunks=17)


def test_first_step_evaluates_every_chunk():
    assert MlChunkBudget(LAYOUT).current == 17


def test_needed_counts_monomers_and_caps_dimers():
    b = MlChunkBudget(LAYOUT)
    assert b.needed(0) == 1
    assert b.needed(256 - 181) == 1
    assert b.needed(256 - 181 + 1) == 2
    assert b.needed(1700) == -(-(181 + 1700) // 256)
    assert b.needed(10**6) == 17  # beyond the cap: capped, never more than all chunks


def test_update_shrinks_with_headroom_and_hysteresis():
    b = MlChunkBudget(LAYOUT)
    # 181 + 1700 = 1881 used slots (+32 spare) -> 8 chunks of 256
    assert b.update(1700) == 8
    assert b.covers(1700)
    # one chunk less needed: stays (slack < 2), no recompile
    assert b.update(1700 - 256) == 8
    # spare slots cross into the next chunk: covered now, grows for the next step
    assert b.covers(1850) and b.needed(1850) == 8
    assert b.update(1850) == 9
    # large drop: shrink
    assert b.update(500) == 3


def test_overflow_is_detected():
    b = MlChunkBudget(LAYOUT)
    b.update(1700)
    assert b.current == 8
    assert b.covers(8 * 256 - 181)
    assert not b.covers(8 * 256 - 181 + 1)


def test_saturation_raises_instead_of_warning():
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        SparseDimerCapOverflow,
    )

    b = MlChunkBudget(LAYOUT)
    b.update(1700)
    with pytest.raises(SparseDimerCapOverflow) as exc:
        b.raise_if_saturated(4100)
    assert exc.value.n_active == 4100
    assert exc.value.cap == 4005
    assert exc.value.dropped == 95
    # Chunk `covers` is about PhysNet chunks, not the dimer cap. Overflow is a
    # separate fail-closed check; growing chunks cannot recover dropped pairs.
    with pytest.raises(SparseDimerCapOverflow):
        b.note_saturation(4100)


def test_callback_raises_on_cap_overflow():
    from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
        SparseDimerCapOverflow,
    )

    budget = MlChunkBudget(LAYOUT)

    def forward(*args, ml_eval_chunks=None):
        return jnp.float64(1.0), jnp.ones((2, 3)), jnp.int32(4100)

    with pytest.raises(SparseDimerCapOverflow):
        _FakeCalc()._check_ml_chunk_budget(budget, forward, (), forward())


class _FakeCalc:
    _check_ml_chunk_budget = DecomposedMlpotCalculator._check_ml_chunk_budget


def test_callback_reruns_when_budget_too_small():
    budget = MlChunkBudget(LAYOUT)
    budget.update(1000)  # 1181 + 32 slots -> 5 chunks
    assert budget.current == 5
    n_active = 1700  # needs 8 chunks
    calls = []

    def forward(*args, ml_eval_chunks=None):
        calls.append(ml_eval_chunks)
        return jnp.float64(ml_eval_chunks or 0), jnp.full((2, 3), ml_eval_chunks or -1), jnp.int32(n_active)

    calc = _FakeCalc()
    first = forward("x")
    e, f = calc._check_ml_chunk_budget(budget, forward, ("x",), first)
    assert calls == [None, 8]  # re-ran once with the grown budget
    assert float(e) == 8.0
    np.testing.assert_array_equal(np.asarray(f), np.full((2, 3), 8))
    assert budget.covers(n_active) and budget.current == 8
    assert calc._ml_chunk_budget_reruns == 1


def test_callback_keeps_result_when_budget_covers():
    budget = MlChunkBudget(LAYOUT)
    calls = []

    def forward(*args, ml_eval_chunks=None):
        calls.append(ml_eval_chunks)
        return jnp.float64(1.0), jnp.ones((2, 3)), jnp.int32(1700)

    e, f = _FakeCalc()._check_ml_chunk_budget(budget, forward, (), forward())
    assert calls == [None]
    assert float(e) == 1.0 and np.all(np.asarray(f) == 1.0)
    assert budget.current == 8


def test_dense_batch_count_disables_check():
    budget = MlChunkBudget(LAYOUT)

    def forward(*args, ml_eval_chunks=None):
        return jnp.float64(2.0), jnp.ones((2, 3)), jnp.int32(-1)

    e, _ = _FakeCalc()._check_ml_chunk_budget(budget, forward, (), forward())
    assert float(e) == 2.0 and budget.current == 17
