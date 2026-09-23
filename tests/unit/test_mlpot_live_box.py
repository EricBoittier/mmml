"""The cached MLpot forward must use the box of the calculator running the callback.

The jitted forward is cached on the model (``_grad_cache_owner``) and shared by every
calculator it registers. It used to read ``_current_box`` from the calculator that built
it, so after a re-registration (``refresh_mlpot_energy_and_grms``, CPT sub-chunks) it
kept evaluating in the first calculator's cell.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator


def _box_energy_fn(**kw):
    """Energy = trace(box), so the forward's box is visible in its energy and dE/dbox."""
    return SimpleNamespace(energy=jnp.trace(kw["box"]), forces=jnp.zeros_like(kw["positions"]))


def _calculator(owner):
    calc = DecomposedMlpotCalculator(_box_energy_fn, CutoffParameters(), 2, np.ones(8, dtype=int), cell=20.0, do_mm=False)
    calc._parent_model = owner
    return calc


def _args():
    empty = jnp.zeros((0,), dtype=jnp.int32)
    return (jnp.zeros((8, 3)), None, None, False, empty, empty, False)


def test_reregistered_calculator_box_reaches_cached_forward() -> None:
    owner = SimpleNamespace(_forward_cache_key=None, _spherical_forward_fn=None)
    first, second = _calculator(owner), _calculator(owner)
    z = jnp.ones(8, dtype=jnp.int32)

    first._set_live_callback_box(jnp.eye(3) * 20.0)
    fwd = first._get_spherical_forward_fn(n_atoms=8, atomic_numbers_jax=z, box_jax=jnp.eye(3) * 20.0)
    assert float(fwd(*_args())[0]) == 60.0

    # A new registration builds a new calculator; the forward comes from the owner's cache.
    second._set_live_callback_box(jnp.eye(3) * 25.0)
    fwd2 = second._get_spherical_forward_fn(n_atoms=8, atomic_numbers_jax=z, box_jax=jnp.eye(3) * 20.0)
    assert fwd2 is fwd
    assert float(fwd2(*_args())[0]) == 75.0

    e, _f, _n, dE_dbox = owner._spherical_forward_vir_fn(*_args())
    assert float(e) == 75.0
    np.testing.assert_allclose(np.asarray(dE_dbox), np.eye(3))
