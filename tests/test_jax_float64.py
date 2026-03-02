"""
Tests for JAX float64 (jax_enable_x64) support.

JAX uses jax_enable_x64 (default False) to control 64-bit types. When False,
float64 is truncated to float32. These tests verify the mechanism works.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_in_subprocess(code: str, env: dict | None = None) -> tuple[int, str]:
    """Run Python code in a fresh subprocess. Returns (returncode, combined output)."""
    env = dict(os.environ) if env is None else {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode, out


def test_jax_default_is_float32():
    """By default JAX uses float32; float64 is truncated when x64 disabled."""
    code = """
import warnings
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    import jax.numpy as jnp
    # Default arrays are float32 when x64 disabled
    y = jnp.array([1.0])
    assert y.dtype == jnp.float32, f"default should be float32, got {y.dtype}"
    # Explicit float64 is truncated to float32 when x64 disabled
    x = jnp.array([1.0], dtype=jnp.float64)
    assert x.dtype == jnp.float32, f"float64 truncated to float32 when x64 off, got {x.dtype}"
print("OK")
"""
    ret, out = _run_in_subprocess(code)
    assert ret == 0, f"Subprocess failed: {out}"
    assert "OK" in out


def test_jax_enable_x64_via_env():
    """JAX_ENABLE_X64=True allows float64 arrays."""
    code = """
import jax.numpy as jnp
x = jnp.array([1.0], dtype=jnp.float64)
assert x.dtype == jnp.float64, f"expected float64, got {x.dtype}"
# Default scalar promotion also uses float64 when x64 enabled
y = jnp.array(1.0)
assert y.dtype == jnp.float64, f"expected float64 default, got {y.dtype}"
print("OK")
"""
    ret, out = _run_in_subprocess(code, env={"JAX_ENABLE_X64": "true"})
    assert ret == 0, f"Subprocess failed: {out}"
    assert "OK" in out


def test_jax_enable_x64_via_config():
    """jax.config.update('jax_enable_x64', True) enables float64."""
    code = """
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
x = jnp.array([1.0], dtype=jnp.float64)
assert x.dtype == jnp.float64, f"expected float64, got {x.dtype}"
print("OK")
"""
    ret, out = _run_in_subprocess(code)
    assert ret == 0, f"Subprocess failed: {out}"
    assert "OK" in out


def test_jax_float64_used_with_json_params():
    """json_to_params + JAX backend with float64 produces float64 JAX arrays."""
    code = """
import json
import tempfile
from pathlib import Path

# Must set x64 BEFORE importing jax.numpy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mmml.utils.model_checkpoint import json_to_params, to_jsonable

params = {"x": [[0.1, 0.2], [0.3, 0.4]]}
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({"params": params}, f)
    path = f.name

try:
    loaded = json_to_params(path, dtype="float64", backend="jax")
    arr = loaded["params"]["x"]
    assert isinstance(arr, jnp.ndarray), f"expected jax array, got {type(arr)}"
    assert arr.dtype == jnp.float64, f"expected float64, got {arr.dtype}"
    print("OK")
finally:
    Path(path).unlink(missing_ok=True)
"""
    ret, out = _run_in_subprocess(code)
    assert ret == 0, f"Subprocess failed: {out}"
    assert "OK" in out
