"""Unit tests for ML compute dtype resolution."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.ml_dtypes import (
    add_ml_compute_dtype_args,
    as_ml_array,
    cast_pytree_to_ml_dtype,
    json_tree_to_jax_params,
    ml_numpy_dtype,
    ml_scalar,
    ml_zeros,
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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float32", jnp.float32),
        ("f32", jnp.float32),
        ("32", jnp.float32),
        ("float64", jnp.float64),
        ("f64", jnp.float64),
        ("64", jnp.float64),
    ],
)
def test_resolve_ml_compute_dtype_aliases(monkeypatch, name, expected):
    monkeypatch.delenv("MMML_ML_DTYPE", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    got = resolve_ml_compute_dtype(name)
    if expected == jnp.float64 and not jax.config.read("jax_enable_x64"):
        assert got == jnp.float32
    else:
        assert got == expected


def test_ml_numpy_dtype_maps_jax_to_numpy():
    assert ml_numpy_dtype(jnp.float32) == np.dtype("float32")
    assert ml_numpy_dtype(jnp.float64) == np.dtype("float64")


def test_as_ml_array_ml_scalar_ml_zeros_use_resolved_dtype(monkeypatch):
    monkeypatch.delenv("MMML_ML_DTYPE", raising=False)
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    target = resolve_ml_compute_dtype()
    assert as_ml_array([1.0, 2.0]).dtype == target
    assert ml_scalar(3.5).dtype == target
    assert ml_zeros((2, 3)).dtype == target


def test_as_ml_array_explicit_dtype_overrides_resolution(monkeypatch):
    monkeypatch.setenv("MMML_ML_DTYPE", "float64")
    assert as_ml_array([1.0], dtype=jnp.float32).dtype == jnp.float32


def test_json_tree_to_jax_params_nested_structure():
    tree = {
        "layer": {
            "w": [[1.0, 2.0]],
            "tags": ["a", "b"],
        }
    }
    out = json_tree_to_jax_params(tree, dtype=jnp.float32)
    assert out["layer"]["w"].dtype == jnp.float32
    assert out["layer"]["tags"] == ["a", "b"]


def test_add_ml_compute_dtype_args_registers_flag():
    import argparse

    parser = argparse.ArgumentParser()
    add_ml_compute_dtype_args(parser)
    args = parser.parse_args(["--ml-compute-dtype", "float64"])
    assert args.ml_compute_dtype == "float64"
