"""--fixed-cgenff-vdw must pin the CGenFF LJ prior at its published parameters.

By default the model may rescale the LJ epsilon through three learned paths: a
network-predicted per-atom scale, a global scale, and a per-element scale. A trained
checkpoint drove these to global=0.14 with element scales of 0.10 (C) / 0.24 (H) --
carbon-carbon epsilon at ~1.4% of its physical value, i.e. the force-field prior was
effectively erased and the neural term took over. This flag removes those degrees of
freedom so the prior can only be corrected, never scaled away.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_so3lr_spooky_extxyz.py"


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location("train_so3lr_spooky_extxyz", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # flax's dataclass transform resolves the defining module via sys.modules, so the
    # module must be registered before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        features=32, max_degree=1, num_iterations=1, num_basis_functions=8, cutoff=6.0,
        max_atomic_number=18, predict_charges=True, n_res=1, no_zbl=False,
        trainable_zbl=False, zbl_cuton=0.1, zbl_cutoff=0.6, efa=False,
        use_energy_bias=False, electrostatics_damping_sigma=4.0,
        fixed_cgenff_vdw=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_flag_sets_model_attributes(trainer):
    free = trainer.create_model(_args(fixed_cgenff_vdw=False), max_atoms=8)
    assert free.learn_cgenff_vdw_scale is True
    assert free.predict_atomic_vdw_scale is True

    fixed = trainer.create_model(_args(fixed_cgenff_vdw=True), max_atoms=8)
    assert fixed.learn_cgenff_vdw_scale is False
    assert fixed.predict_atomic_vdw_scale is False


def _vdw_scale_leaves(params) -> list[str]:
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return [
        "/".join(str(k.key) for k in path if hasattr(k, "key"))
        for path, _ in flat
        if any("vdw_scale" in str(getattr(k, "key", "")) for k in path)
    ]


def test_fixed_prior_has_no_vdw_scale_parameters(trainer):
    """The load-bearing check: with the flag on, there is no parameter that can rescale
    the LJ prior, so it cannot be trained away."""
    n = 6
    model = trainer.create_model(_args(fixed_cgenff_vdw=True), max_atoms=n)
    key = jax.random.PRNGKey(0)
    params = model.init(
        key,
        atomic_numbers=jnp.array([8, 1, 1, 8, 1, 1]),
        positions=jnp.zeros((n, 3)),
        dst_idx=jnp.array([0, 1]),
        src_idx=jnp.array([1, 0]),
        batch_segments=jnp.zeros(n, dtype=jnp.int32),
        batch_size=1,
        batch_mask=jnp.ones(2),
        atom_mask=jnp.ones(n),
        charges=jnp.zeros(n),
        spins=jnp.zeros(n),
    )
    assert _vdw_scale_leaves(params) == []
