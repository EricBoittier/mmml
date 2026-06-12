"""Regression tests for the PhysNet JAX model restored from portable JSON params.

These tests exercise the JSON checkpoint path
(``examples/ckpts_json/DESdimers_params.json``) directly, without an Orbax
checkpoint directory, an NPZ dataset, CHARMM, or a GPU. They run on CPU so they
are safe for CI.

The reference energy/force values below were captured on CPU (float32). They act
as a regression guard: if the model code, the params file, or the JAX/e3x
versions change the numerics, these tests fail and the references must be updated
deliberately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_PARAMS = PROJECT_ROOT / "examples/ckpts_json/DESdimers_params.json"

# Fixed acetone-like geometry (10 atoms: C3 O H6), in Angstrom.
_Z = np.array([6, 6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
_R = np.array(
    [
        [0.00, 0.00, 0.00],
        [1.52, 0.00, 0.00],
        [-0.76, 1.30, 0.00],
        [1.22, -1.15, 0.20],
        [2.10, 0.50, 0.80],
        [2.10, 0.30, -0.90],
        [1.60, -0.95, 0.10],
        [-1.80, 1.05, 0.10],
        [-0.55, 1.90, 0.88],
        [-0.55, 1.95, -0.85],
    ],
    dtype=np.float64,
)

# Reference values captured on CPU (float32) from DESdimers_params.json.
_REF_ENERGY = -40.14439392089844
_REF_FORCE_NORM = 48.72711944580078
_REF_FORCE_ATOM0 = np.array([-8.877176, 12.343563, -1.0513406])


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _skip_reason() -> str | None:
    if not JSON_PARAMS.exists():
        return f"JSON params not found: {JSON_PARAMS}"
    for mod in ("jax", "e3x", "flax"):
        if not _can_import(mod):
            return f"{mod} not available in this environment"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


def _build_model_and_params():
    """Restore the EF model + params from the JSON checkpoint."""
    from mmml.utils.model_checkpoint import json_to_params
    from mmml.models.physnetjax.physnetjax.models.model import EF

    loaded = json_to_params(JSON_PARAMS, backend="jax")
    cfg = loaded["config"]
    params = loaded["params"]
    attrs = EF().return_attributes()
    model = EF(**{k: cfg[k] for k in attrs if k in cfg})
    return model, params


def _evaluate():
    """Evaluate energy + forces on the fixed geometry, pinned to the CPU device."""
    import jax
    import jax.numpy as jnp
    import e3x

    model, params = _build_model_and_params()
    n = len(_Z)
    with jax.default_device(jax.devices("cpu")[0]):
        dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(n)
        out = model.apply(
            params,
            atomic_numbers=jnp.asarray(_Z),
            positions=jnp.asarray(_R),
            dst_idx=dst_idx,
            src_idx=src_idx,
        )
        energy = float(np.asarray(out["energy"]).squeeze())
        forces = np.asarray(out["forces"], dtype=np.float64)
    return energy, forces


def test_json_params_load_and_eval_finite():
    energy, forces = _evaluate()
    assert np.isfinite(energy)
    assert forces.shape == (len(_Z), 3)
    assert np.all(np.isfinite(forces))


def test_json_params_energy_regression():
    energy, _ = _evaluate()
    np.testing.assert_allclose(energy, _REF_ENERGY, rtol=2e-3, atol=5e-2)


def test_json_params_forces_regression():
    _, forces = _evaluate()
    np.testing.assert_allclose(
        np.linalg.norm(forces), _REF_FORCE_NORM, rtol=2e-3, atol=5e-2
    )
    np.testing.assert_allclose(forces[0], _REF_FORCE_ATOM0, rtol=2e-3, atol=2e-2)


def test_json_params_eval_is_deterministic():
    e1, f1 = _evaluate()
    e2, f2 = _evaluate()
    np.testing.assert_allclose(e1, e2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f1, f2, rtol=1e-6, atol=1e-6)
