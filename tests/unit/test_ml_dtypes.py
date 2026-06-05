"""Unit tests for ML compute dtype resolution."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.ml_dtypes import (
    cast_pytree_to_ml_dtype,
    json_tree_to_jax_params,
    resolve_ml_compute_dtype,
)


def test_resolve_ml_compute_dtype_defaults_to_float32(monkeypatch):
    monkeypatch.delenv("MMML_ML_DTYPE", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    assert resolve_ml_compute_dtype() == jnp.float32
    assert resolve_ml_compute_dtype(None) == jnp.float32


def test_resolve_ml_compute_dtype_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("MMML_ML_DTYPE", "float64")
    assert resolve_ml_compute_dtype("float32") == jnp.float32


def test_resolve_ml_compute_dtype_from_env(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.setenv("MMML_ML_DTYPE", "float64")
    if jax.config.read("jax_enable_x64"):
        assert resolve_ml_compute_dtype() == jnp.float64
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_ml_compute_dtype() == jnp.float32
        assert any("jax_enable_x64 is False" in str(w.message) for w in caught)


def test_resolve_ml_compute_dtype_jax_enable_x64_env(monkeypatch):
    monkeypatch.delenv("MMML_ML_DTYPE", raising=False)
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    if jax.config.read("jax_enable_x64"):
        assert resolve_ml_compute_dtype() == jnp.float64
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_ml_compute_dtype() == jnp.float32
        assert caught


def test_resolve_ml_compute_dtype_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported ML compute dtype"):
        resolve_ml_compute_dtype("float16")


def test_json_tree_to_jax_params_promotes_floats(monkeypatch):
    monkeypatch.delenv("MMML_ML_DTYPE", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    tree = {"w": [[1.0, 2.0], [3.0, 4.0]], "n": 7}
    out = json_tree_to_jax_params(tree, dtype=jnp.float64)
    expected = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    assert out["w"].dtype == expected
    assert out["n"] == 7


def test_cast_pytree_to_ml_dtype_only_float_leaves():
    tree = {
        "w": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3], dtype=np.int32),
    }
    out = cast_pytree_to_ml_dtype(tree, dtype=jnp.float64)
    expected = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    assert out["w"].dtype == expected
    assert out["b"].dtype == jnp.int32
